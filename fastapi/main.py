from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import contextlib
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import traceback


app = FastAPI()

REQUIREMENTS_PATH = Path(os.getenv("REQUIREMENTS_PATH", "/app/requirements.txt"))
PACKAGE_PATTERN = re.compile(
    r"^[A-Za-z0-9_.-]+(\[[A-Za-z0-9_,.-]+\])?([<>=!~]=?[A-Za-z0-9_.*+!<>=~,-]+)?$"
)
requirements_lock = threading.Lock()
PIP_INSTALL_TIMEOUT = 600


class ExecuteRequest(BaseModel):
    code: str
    timeout: int = 30


class InstallPackagesRequest(BaseModel):
    packages: list[str] = Field(min_length=1, max_length=20)


def build_pip_install_command(packages: list[str]) -> tuple[list[str], list[str]]:
    safe_packages = []
    for package in packages:
        package = package.strip()
        if not package or not PACKAGE_PATTERN.fullmatch(package):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported package spec: {package!r}",
            )
        safe_packages.append(package)
    return safe_packages, [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        *safe_packages,
    ]


def read_requirements() -> list[str]:
    if not REQUIREMENTS_PATH.exists():
        return []
    return REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()


def append_missing_requirements(packages: list[str]) -> list[str]:
    with requirements_lock:
        existing_lines = read_requirements()
        normalized_existing = {
            line.strip().lower()
            for line in existing_lines
            if line.strip() and not line.strip().startswith("#")
        }
        missing = [
            package
            for package in packages
            if package.lower() not in normalized_existing
        ]
        if missing:
            separator = "" if not existing_lines or existing_lines[-1] == "" else "\n"
            REQUIREMENTS_PATH.write_text(
                "\n".join(existing_lines) + separator + "\n".join(missing) + "\n",
                encoding="utf-8",
            )
        return missing


def run_command(args: list[str], timeout: int) -> dict:
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": args,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


@app.post("/execute")
async def execute(req: ExecuteRequest):
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    local_vars = {}

    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
            stderr_capture
        ):
            exec(compile(req.code, "<string>", "exec"), local_vars)

        return {
            "success": True,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "result": {
                k: repr(v) for k, v in local_vars.items() if not k.startswith("_")
            },
        }
    except Exception:
        return {
            "success": False,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "error": traceback.format_exc(),
        }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/packages/install")
async def install_packages(req: InstallPackagesRequest):
    packages, command = build_pip_install_command(req.packages)
    install_result = run_command(command, PIP_INSTALL_TIMEOUT)
    if install_result["returncode"] != 0:
        return {
            "success": False,
            "packages": packages,
            "requirements_added": [],
            "install": install_result,
        }

    requirements_added = append_missing_requirements(packages)

    return {
        "success": True,
        "packages": packages,
        "requirements_added": requirements_added,
        "install": install_result,
    }


@app.get("/packages/status")
async def packages_status():
    return {
        "requirements_path": str(REQUIREMENTS_PATH),
        "requirements": read_requirements(),
    }


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
            imported_module = __import__(module)
            results[name] = getattr(imported_module, "__version__", "ok")
        except ImportError as e:
            results[name] = f"ERROR: {e}"
    return results


@app.get("/data/files")
async def list_files():
    files = []
    for filename in os.listdir("/app/data"):
        path = f"/app/data/{filename}"
        files.append(
            {
                "name": filename,
                "size_kb": round(os.path.getsize(path) / 1024, 1),
            }
        )
    return {"files": files}
