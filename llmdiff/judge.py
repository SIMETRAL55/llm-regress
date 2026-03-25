import logging
import re
import time
from datetime import datetime

import litellm

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """\
You are evaluating two AI-generated responses for a single quality criterion.

Input: {input}

Criterion: {criterion}

Response A (v1):
{output_v1}

Response B (v2):
{output_v2}

Which response better satisfies the criterion? Reply with exactly one of: A, B, or tie
Then on the next line, one sentence explaining your reasoning.
"""


def _parse_verdict(text: str) -> tuple[str, str]:
    """Extract (verdict, reasoning) from a judge model response."""
    lines = text.strip().splitlines()
    verdict = "tie"
    reasoning = ""

    for line in lines:
        token = line.strip().lower()
        if token in ("a", "b", "tie"):
            verdict = token
            break
        # Handle cases like "A." or "**A**" or "verdict: A"
        match = re.search(r"\b(a|b|tie)\b", token)
        if match:
            verdict = match.group(1)
            break

    # Reasoning is everything after the first verdict line
    for i, line in enumerate(lines):
        if re.search(r"\b(a|b|tie)\b", line.strip().lower()):
            reasoning = " ".join(lines[i + 1:]).strip()
            break

    return verdict, reasoning or "No reasoning provided."


def judge_pair(
    input: str,
    output_v1: str,
    output_v2: str,
    criterion: str,
    judge_model: str,
) -> dict:
    """Judge a single criterion between two outputs using 3x majority vote.

    Returns a dict matching the criteria_results shape in CLAUDE.md.
    """
    prompt = JUDGE_PROMPT.format(
        input=input,
        criterion=criterion,
        output_v1=output_v1,
        output_v2=output_v2,
    )

    verdicts = []
    reasoning_samples = []

    for _ in range(3):
        for attempt in range(4):  # up to 4 attempts with backoff
            try:
                response = litellm.completion(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.choices[0].message.content
                verdict, reasoning = _parse_verdict(text)
                verdicts.append(verdict)
                reasoning_samples.append(reasoning)
                time.sleep(2)  # stay under 30 RPM free tier limit
                break
            except litellm.RateLimitError:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s
                logger.warning(f"Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            except Exception as e:
                logger.error(f"Judge call failed: {e}")
                verdicts.append("tie")
                reasoning_samples.append("Judge call failed.")
                break
        else:
            # All retry attempts exhausted
            verdicts.append("tie")
            reasoning_samples.append("Judge call failed after retries.")

    # Majority vote
    counts = {"a": verdicts.count("a"), "b": verdicts.count("b"), "tie": verdicts.count("tie")}
    majority = max(counts, key=counts.get)

    if counts["a"] == counts["b"] == counts["tie"]:
        majority = "tie"

    # Confidence
    top_count = counts[majority]
    if top_count == 3:
        confidence = "high"
    elif top_count == 2:
        confidence = "medium"
    else:
        confidence = "low"

    # Map verdict
    verdict_map = {"a": "regressed", "b": "improved", "tie": "neutral"}
    verdict = verdict_map[majority]

    # Scores: winner=1.0, loser=0.0, tie=0.5 each
    if majority == "a":
        score_v1, score_v2 = 1.0, 0.0
    elif majority == "b":
        score_v1, score_v2 = 0.0, 1.0
    else:
        score_v1, score_v2 = 0.5, 0.5

    delta = round(score_v2 - score_v1, 4)

    # Use the reasoning from the majority-agreeing call, or the first one
    majority_idx = next(
        (i for i, v in enumerate(verdicts) if v == majority),
        0,
    )
    reasoning = reasoning_samples[majority_idx]

    return {
        "criterion": criterion,
        "score_v1": score_v1,
        "score_v2": score_v2,
        "delta": delta,
        "verdict": verdict,
        "reasoning": reasoning,
        "confidence": confidence,
    }


def judge_test_case(runner_result: dict, criteria: list[str], judge_model: str) -> dict:
    """Judge all criteria for one test case.

    Returns the full test case result shape with criteria_results and overall_verdict.
    """
    criteria_results = []
    for criterion in criteria:
        result = judge_pair(
            input=runner_result["input"],
            output_v1=runner_result["output_v1"],
            output_v2=runner_result["output_v2"],
            criterion=criterion,
            judge_model=judge_model,
        )
        criteria_results.append(result)

    if criteria_results:
        avg_delta = sum(r["delta"] for r in criteria_results) / len(criteria_results)
    else:
        avg_delta = 0.0

    if avg_delta > 0.1:
        overall_verdict = "improved"
    elif avg_delta < -0.1:
        overall_verdict = "regressed"
    else:
        overall_verdict = "neutral"

    return {
        "id": runner_result["id"],
        "input": runner_result["input"],
        "output_v1": runner_result["output_v1"],
        "output_v2": runner_result["output_v2"],
        "criteria_results": criteria_results,
        "overall_verdict": overall_verdict,
    }


def judge_run(runner_results: list[dict], config: dict) -> dict:
    """Judge all test cases in a run and return the full run result shape."""
    run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now().isoformat()
    judge_model = config.get("judge_model", config.get("model"))

    # Build a lookup: test case id → criteria list
    tc_criteria = {tc["id"]: tc.get("criteria", []) for tc in config.get("test_cases", [])}

    judged_cases = []
    for result in runner_results:
        criteria = tc_criteria.get(result["id"], [])
        judged = judge_test_case(result, criteria, judge_model)
        judged_cases.append(judged)

    # Summary
    verdicts = [tc["overall_verdict"] for tc in judged_cases]
    improved = verdicts.count("improved")
    regressed = verdicts.count("regressed")
    neutral = verdicts.count("neutral")

    all_deltas = [
        cr["delta"]
        for tc in judged_cases
        for cr in tc["criteria_results"]
    ]
    score_delta_avg = round(sum(all_deltas) / len(all_deltas), 4) if all_deltas else 0.0

    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "yaml_file": config.get("yaml_file", ""),
        "model": config.get("model", ""),
        "test_cases": judged_cases,
        "summary": {
            "total": len(judged_cases),
            "improved": improved,
            "regressed": regressed,
            "neutral": neutral,
            "score_delta_avg": score_delta_avg,
        },
    }
