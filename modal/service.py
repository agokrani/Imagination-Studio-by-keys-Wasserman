"""Authenticated, durable HTTP access to the guarded H3 Modal runner."""
import datetime
import hmac
import os
from pathlib import Path
import re
import secrets
import sys
import uuid

import modal
sys.path.insert(0, str(Path(__file__).resolve().parent if modal.is_local() else Path("/opt/studio-modal")))
import h3
from service_support import video_config

app = modal.App("studio-h3-service")
jobs = modal.Dict.from_name("studio-h3-service-jobs", create_if_missing=True)
access = modal.Secret.from_name("studio-h3-service")
web_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "fastapi==0.142.2", "python-multipart==0.0.32").env(h3.volume_env)
if modal.is_local():
    web_image = web_image.add_local_dir(h3.HERE, "/opt/studio-modal", ignore=["__pycache__", "*.pyc", "examples"])


@app.function(image=h3.download_image, volumes={"/weights": h3.weights}, cpu=4, memory=8192,
              timeout=3600, max_containers=1, retries=0)
def prepare(mode: str):
    return h3.prepare.local(mode)


@app.function(image=h3.image, volumes={"/weights": h3.weights, "/renders": h3.outputs}, gpu=["B200", "H200"],
              cpu=(4, 4), memory=(262144, 262144), timeout=86400, max_containers=1,
              single_use_containers=True, retries=0)
def render(job_id: str, config: dict, run_id: str):
    job = jobs["job:" + job_id]
    jobs["job:" + job_id] = {**job, "status": "running"}
    try:
        h3.outputs.reload()
        names = ([config["first_frame"]] if "first_frame" in config else []) + sum(
            [config.get(field, []) for field in ("images", "videos", "audios")], [])
        files = {name: (Path("/renders/uploads") / name).read_bytes() for name in names}
        result = h3.render.local(config, files, run_id)
        jobs["job:" + job_id] = {**job, "status": "done", "outputs": [f"renders/{job_id}.mp4"],
                                "evidence": result}
    except BaseException:
        jobs["job:" + job_id] = {**job, "status": "failed", "error": "Render failed; inspect the private Modal logs"}
        raise


@app.function(image=web_image, volumes={"/renders": h3.outputs}, secrets=[access],
              max_containers=1, timeout=300)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse
    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    key = os.environ["STUDIO_ACCESS_KEY"]
    if len(key) < 32:
        raise RuntimeError("A strong Studio access key is required")

    @api.middleware("http")
    async def authenticate(request: Request, call_next):
        supplied = request.headers.get("X-Studio-Key", "")
        if not hmac.compare_digest(supplied.encode(), key.encode()):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)

    @api.get("/health")
    def health():
        return {"status": "ok"}

    @api.post("/api/upload")
    def upload(file: UploadFile):
        extension = Path(file.filename or "").suffix.lower()
        if extension not in {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".wav", ".mp3", ".m4a", ".ogg", ".flac"}:
            raise HTTPException(400, "Unsupported media extension")
        h3.outputs.reload()
        name = uuid.uuid4().hex + extension
        directory = Path("/renders/uploads")
        directory.mkdir(exist_ok=True)
        target = directory / name
        length = 0
        try:
            with target.open("xb") as stream:
                while block := file.file.read(1024 * 1024):
                    length += len(block)
                    if length > 512 * 1024 * 1024:
                        raise HTTPException(413, "Media exceeds 512 MiB")
                    stream.write(block)
            if not length:
                raise HTTPException(400, "Empty media")
            h3.outputs.commit()
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return {"path": "uploads/" + name}

    @api.post("/api/jobs")
    async def submit(request: Request):
        try:
            body = await request.json()
            config = video_config(body)
        except (ValueError, TypeError):
            raise HTTPException(400, "Invalid video request")
        h3.outputs.reload()
        names = ([config["first_frame"]] if "first_frame" in config else []) + sum(
            [config.get(field, []) for field in ("images", "videos", "audios")], [])
        if any(not (Path("/renders/uploads") / name).is_file() for name in names):
            raise HTTPException(400, "Reference unavailable")
        job_id = datetime.datetime.now(datetime.timezone.utc).strftime("%m%d-%H%M%S-") + secrets.token_hex(3)
        run_id = uuid.uuid4().hex
        job = {"id": job_id, "status": "queued", "outputs": [], "run_id": run_id}
        # Save acceptance before spawning. A lost spawn response remains uncertain, never resubmitted.
        jobs["job:" + job_id] = job
        call = render.spawn(job_id, config, run_id)
        jobs["call:" + job_id] = call.object_id
        return job

    @api.get("/api/jobs/{job_id}")
    def status(job_id: str):
        job = jobs.get("job:" + job_id)
        if not job:
            raise HTTPException(404, "Job unavailable")
        if job["status"] in {"queued", "running"}:
            call = jobs.get("call:" + job_id)
            if call:
                try:
                    modal.FunctionCall.from_id(call).get(timeout=0)
                except (TimeoutError, modal.exception.TimeoutError):
                    pass
                except modal.exception.NotFoundError:
                    return {**job, "status": "unknown"}
                except Exception:
                    return {**job, "status": "unknown"}
            job = jobs["job:" + job_id]
        return job

    @api.get("/files/renders/{filename}")
    def download(filename: str):
        if not re.fullmatch(r"\d{4}-\d{6}-[a-f0-9]{6}\.mp4", filename):
            raise HTTPException(404, "Output unavailable")
        job = jobs.get("job:" + filename[:-4])
        if not job or job["status"] != "done":
            raise HTTPException(404, "Output unavailable")
        h3.outputs.reload()
        return FileResponse(Path("/renders") / job["run_id"] / "video.mp4", media_type="video/mp4")

    return api
