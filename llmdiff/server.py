import json
import os
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from llmdiff import storage
from llmdiff.runner import run_test_cases
from llmdiff.judge import judge_test_case

DB_PATH = os.environ.get("LLMDIFF_DB_PATH", "~/.llmdiff/history.db")
PORT = int(os.environ.get("LLMDIFF_PORT", "7331"))

app = FastAPI(title="LLM Diff")

_UI_PATH = Path(__file__).parent / "templates" / "ui.html"


@app.on_event("startup")
def startup():
    storage.init_db(DB_PATH)


@app.get("/", response_class=HTMLResponse)
def index():
    if _UI_PATH.exists():
        return _UI_PATH.read_text()
    return "<h1>LLM Diff</h1><p>UI not built yet.</p>"


@app.get("/api/runs")
def api_list_runs():
    return storage.list_runs(DB_PATH)


@app.get("/api/runs/{run_id}")
def api_get_run(run_id: str):
    try:
        return storage.get_run(run_id, DB_PATH)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")


class StreamRequest(BaseModel):
    yaml_file: str
    db_path: str = DB_PATH


@app.post("/api/runs/stream")
async def api_stream_run(req: StreamRequest):
    async def event_generator():
        try:
            with open(req.yaml_file) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
            return

        config["yaml_file"] = req.yaml_file
        test_cases = config.get("test_cases", [])
        judge_model = config.get("judge_model", config.get("model"))

        from datetime import datetime
        run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = datetime.now().isoformat()

        yield {
            "event": "start",
            "data": json.dumps({"total": len(test_cases), "run_id": run_id}),
        }

        judged_cases = []
        tc_criteria = {tc["id"]: tc.get("criteria", []) for tc in test_cases}

        for tc in test_cases:
            runner_results = run_test_cases({"model": config["model"], "test_cases": [tc]})
            if runner_results:
                result = runner_results[0]
                criteria = tc_criteria.get(result["id"], [])
                judged = judge_test_case(result, criteria, judge_model)
                judged_cases.append(judged)
                yield {"event": "result", "data": json.dumps(judged)}

        verdicts = [tc["overall_verdict"] for tc in judged_cases]
        all_deltas = [
            cr["delta"]
            for tc in judged_cases
            for cr in tc.get("criteria_results", [])
        ]
        score_delta_avg = round(sum(all_deltas) / len(all_deltas), 4) if all_deltas else 0.0

        summary = {
            "total": len(judged_cases),
            "improved": verdicts.count("improved"),
            "regressed": verdicts.count("regressed"),
            "neutral": verdicts.count("neutral"),
            "score_delta_avg": score_delta_avg,
        }

        run_result = {
            "run_id": run_id,
            "timestamp": timestamp,
            "yaml_file": req.yaml_file,
            "model": config.get("model", ""),
            "test_cases": judged_cases,
            "summary": summary,
        }

        storage.save_run(run_result, req.db_path)

        yield {"event": "done", "data": json.dumps(summary)}

    return EventSourceResponse(event_generator())


def start():
    import uvicorn
    uvicorn.run("llmdiff.server:app", host="0.0.0.0", port=PORT, reload=False)
