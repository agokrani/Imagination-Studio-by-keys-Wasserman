"""Run Studio's guarded ComfyUI H3 workflows on Modal; no public web endpoint."""
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

import modal

if modal.is_local() and not os.environ.get("MODAL_PROFILE") and not {"--help", "-h"}.intersection(sys.argv):
    raise ValueError("Set MODAL_PROFILE explicitly to the workspace profile you intend to use")

HERE = Path(__file__).resolve().parent if modal.is_local() else Path("/opt/studio-modal")
sys.path.insert(0, str(HERE))
from h3_support import COMFY_COMMIT, MODEL_REVISION, environment, inspect_output, pack_inputs, selected_models, validate, verify, write_json

ROOT = HERE.parent
MANIFEST = json.loads((HERE / "models.json").read_text())
app = modal.App("studio-h3-batch")
weights = modal.Volume.from_name(os.environ.get("STUDIO_MODELS_VOLUME", "studio-h3-models"), create_if_missing=True)
outputs = modal.Volume.from_name(os.environ.get("STUDIO_OUTPUTS_VOLUME", "studio-h3-outputs"), create_if_missing=True)
volume_env = {name: os.environ.get(name, default) for name, default in (
    ("STUDIO_MODELS_VOLUME", "studio-h3-models"), ("STUDIO_OUTPUTS_VOLUME", "studio-h3-outputs"))}
download_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface-hub==1.33.0").env(volume_env)
image = (
    modal.Image.from_registry("python:3.12-slim-bookworm").env(volume_env)
    .apt_install("git", "curl", "ffmpeg", "procps", "libgl1", "libglib2.0-0", "build-essential")
    .run_commands(
        f"git init /opt/studio-runtime/ComfyUI && cd /opt/studio-runtime/ComfyUI && "
        f"git remote add origin https://github.com/Comfy-Org/ComfyUI.git && "
        f"git fetch --depth 1 origin {COMFY_COMMIT} && git checkout --detach FETCH_HEAD"
    )
    .uv_pip_install("torch==2.14.1", "torchvision==0.29.1", "torchaudio==2.11.1", "transformers==5.18.0")
    .run_commands(
        "/.uv/uv pip install --python /usr/local/bin/python -r /opt/studio-runtime/ComfyUI/requirements.txt "
        "torch==2.14.1 torchvision==0.29.1 torchaudio==2.11.1 transformers==5.18.0",
        "mkdir -p /opt/studio-runtime/venv/bin && ln -s /usr/local/bin/python /opt/studio-runtime/venv/bin/python",
    )
)
if modal.is_local():
    download_image = download_image.add_local_dir(HERE, "/opt/studio-modal", ignore=["__pycache__", "*.pyc", "examples"])
    image = (image
             .add_local_dir(HERE, "/opt/studio-modal", ignore=["__pycache__", "*.pyc", "examples"])
             .add_local_dir(ROOT / "spark", "/opt/studio/spark", ignore=["__pycache__", "*.pyc"])
             .add_local_file(ROOT / "scripts/quiet-prompt.sh", "/opt/studio/scripts/quiet-prompt.sh")
             .add_local_file(ROOT / "app/negation.py", "/opt/studio/app/negation.py"))


@app.function(image=download_image, volumes={"/weights": weights}, cpu=4, memory=8192,
              timeout=3600, max_containers=1, retries=0)
def prepare(mode: str):
    from huggingface_hub import hf_hub_download

    entries = selected_models(MANIFEST, mode)
    for entry in entries:
        target = Path("/weights") / entry["directory"] / entry["file"]
        print("Verifying" if target.exists() else "Downloading", entry["file"], flush=True)
        if not target.exists():
            hf_hub_download(MANIFEST["repository"], entry["file"], revision=MODEL_REVISION,
                            local_dir=str(Path("/weights") / entry["directory"]))
        verify(target, entry)
        weights.commit()
    result = {"revision": MODEL_REVISION, "files": entries}
    write_json(Path("/weights") / f"verified-{mode}.json", result)
    weights.commit()
    return result


@app.function(image=image, volumes={"/weights": weights, "/renders": outputs}, gpu=["B200", "H200"],
              cpu=(4, 4), memory=(262144, 262144), timeout=86400, max_containers=1,
              single_use_containers=True, retries=0)
def render(config: dict, files: dict[str, bytes], run_id: str):
    import shutil
    import subprocess

    validate(config)
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("Invalid run ID")
    expected = {"revision": MODEL_REVISION, "files": selected_models(MANIFEST, config["mode"])}
    marker = Path("/weights") / f"verified-{config['mode']}.json"
    if not marker.exists() or json.loads(marker.read_text()) != expected:
        raise RuntimeError("Run prepare for this mode before allocating another GPU")
    for entry in expected["files"]:
        path = Path("/weights") / entry["directory"] / entry["file"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise RuntimeError("Missing or truncated model; rerun prepare")
    root = Path("/renders") / run_id
    root.mkdir(exist_ok=False)
    inputs = root / "inputs"
    inputs.mkdir()
    for name, content in files.items():
        if Path(name).name != name or not content:
            raise ValueError("Inputs must have nonempty content and a basename")
        (inputs / name).write_bytes(content)
    env = environment(config, inputs)
    env["STUDIO_HOME"] = str(root)
    model_paths = "studio:\n  base_path: /weights\n  diffusion_models: |\n    MiniMax-H3-Comfy/diffusion_models\n  text_encoders: |\n    FastH3-Comfy/text_encoders\n  vae: |\n    FastH3-Comfy/vae\n  loras: |\n    MiniMax-H3-Comfy/loras\n"
    Path("/opt/studio-runtime/ComfyUI/extra_model_paths.yaml").write_text(model_paths)
    command = ["bash", "/opt/studio/spark/video.sh", config["prompt"], str(root / "video.mp4")]
    result = {"run_id": run_id, "state": "running", "config": config, "model_revision": MODEL_REVISION,
              "comfy_commit": COMFY_COMMIT, "steps": int(env["STEPS"]), "decoder": env["SPARK_VAE"],
              "device": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                                 text=True, timeout=10).strip()}
    write_json(root / "receipt.json", result)
    outputs.commit()
    started = time.monotonic()
    try:
        with (root / "render.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                print(f"H3 render {run_id}: active ({int(time.monotonic() - started)} seconds)", flush=True)
                time.sleep(30)
            if process.returncode:
                raise RuntimeError(f"Renderer exited {process.returncode}; see {run_id}/render.log in studio-h3-outputs")
        probe = inspect_output(root / "video.mp4", config)
        result.update(state="completed", probe=probe)
    except BaseException as error:
        result.update(state="failed", error=str(error))
        raise
    finally:
        try:
            subprocess.run(["bash", "/opt/studio/spark/comfy.sh", "stop"], env=env, timeout=30, check=False)
        finally:
            comfy_log = Path("/opt/studio-runtime/comfy.log")
            if comfy_log.exists():
                shutil.copyfile(comfy_log, root / "comfy.log")
            result["seconds"] = time.monotonic() - started
            write_json(root / "receipt.json", result)
            outputs.commit()
    return result


@app.local_entrypoint()
def main(phase: str = "prepare", mode: str = "first-frame", config: str = "", output: str = "outputs/modal-h3"):
    if phase == "prepare":
        print(json.dumps(prepare.remote(mode), indent=2))
        return
    directory = Path(output)
    receipt = directory / "submission.json"
    if phase == "render":
        source = Path(config).resolve()
        packed, files = pack_inputs(json.loads(source.read_text()), source.parent)
        directory.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents an accidental resubmission with the same output directory.
        run_id = uuid.uuid4().hex
        with receipt.open("x") as stream:
            json.dump({"state": "submitting", "run_id": run_id}, stream)
        call = render.spawn(packed, files, run_id)
        write_json(receipt, {"state": "submitted", "run_id": run_id, "call_id": call.object_id})
        print(f"Submitted {run_id}; receipt: {receipt}", flush=True)
    elif phase == "collect":
        saved = json.loads(receipt.read_text())
        if "call_id" not in saved:
            raise RuntimeError("Submission outcome is uncertain. Inspect Modal calls and output volume before retrying.")
        call = modal.FunctionCall.from_id(saved["call_id"])
    else:
        raise ValueError("Choose prepare, render or collect")
    result = call.get()
    write_json(directory / "receipt.json", result)
    for name in ("video.mp4", "render.log", "comfy.log"):
        with (directory / name).open("wb") as stream:
            for block in outputs.read_file(f"{result['run_id']}/{name}"):
                stream.write(block)
    print(f"Saved {directory / 'video.mp4'}", flush=True)
