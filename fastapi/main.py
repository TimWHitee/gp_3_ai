from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import contextlib
import io
import traceback
from pydantic import BaseModel

app = FastAPI()



class ExecuteRequest(BaseModel):
    code: str
    timeout: int = 30  # секунды (пока не используется, для документации)

@app.post("/execute")
async def execute(req: ExecuteRequest):
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    local_vars = {}

    try:
        with contextlib.redirect_stdout(stdout_capture), \
             contextlib.redirect_stderr(stderr_capture):
            exec(compile(req.code, "<string>", "exec"), local_vars)

        return {
            "success": True,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "result": {
                k: repr(v)
                for k, v in local_vars.items()
                if not k.startswith("_")
            }
        }
    except Exception:
        return {
            "success": False,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "error": traceback.format_exc()
        }



@app.get("/health")
async def health():
    return {"status": "ok"}



@app.get("/check-imports")
async def check_imports():
    results = {}
    libs = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("catboost", "catboost"),
    ]
    for name, module in libs:
        try:
            m = __import__(module)
            results[name] = getattr(m, "__version__", "ok")
        except ImportError as e:
            results[name] = f"ERROR: {e}"
    return results

import os

@app.get("/data/files")
async def list_files():
    files = []
    for f in os.listdir("/app/data"):
        path = f"/app/data/{f}"
        files.append({
            "name": f,
            "size_kb": round(os.path.getsize(path) / 1024, 1)
        })
    return {"files": files}