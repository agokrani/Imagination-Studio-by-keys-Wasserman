"""Local validation and file handling for the Modal H3 batch runner."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

MODES = ("first-frame", "reference", "reference-turbo")
COMFY_COMMIT = "b9b290e43f1eb379946dfeb013faf9bbefa1a618"
MODEL_REVISION = "912b6e38b2092ec4d2e86d24183e0d59ea059d48"


def validate(config):
    allowed = {"mode", "prompt", "width", "height", "frames", "seed", "first_frame", "images", "videos", "audios"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Config must be an object containing only documented fields")
    mode = config.get("mode")
    if mode not in MODES:
        raise ValueError("Choose first-frame, reference or reference-turbo")
    if not isinstance(config.get("prompt"), str) or not config["prompt"].strip():
        raise ValueError("A nonempty prompt is required")
    for field in ("width", "height"):
        value = config.get(field)
        if type(value) is not int or not 256 <= value <= 1536 or value % 32:
            raise ValueError("Dimensions must be multiples of 32 between 256 and 1536")
    if config["width"] * config["height"] > 768 * 1344:
        raise ValueError("Canvas exceeds 768 × 1344 pixels")
    frames = config.get("frames")
    if type(frames) is not int or not 22 <= frames <= 362 or (frames - 5) % 17:
        raise ValueError("Frames must be 17n + 5, from 22 to 362, at 24 fps")
    if type(config.get("seed")) is not int or not 0 <= config["seed"] <= 2147483647:
        raise ValueError("Seed must be an integer from 0 to 2147483647")
    for field, limit in (("images", 9), ("videos", 3), ("audios", 3)):
        items = config.get(field, [])
        if not isinstance(items, list) or len(items) > limit or any(not isinstance(x, str) or not x for x in items):
            raise ValueError(f"Invalid {field} list (maximum {limit})")
    count = sum(len(config.get(field, [])) for field in ("images", "videos", "audios"))
    if count > 12:
        raise ValueError("At most 12 reference files in total")
    if mode == "first-frame":
        if not isinstance(config.get("first_frame"), str) or not config["first_frame"] or count:
            raise ValueError("First-frame mode needs first_frame and cannot take references or reference audio")
    elif "first_frame" in config or not (config.get("images") or config.get("videos")):
        raise ValueError("Reference mode needs an image or video and cannot take first_frame")
    return config


def pack_inputs(config, directory):
    """Resolve user paths locally; send only generated basenames into the container."""
    validate(config)
    packed, files = dict(config), {}
    for field in ("first_frame", "images", "videos", "audios"):
        names = []
        paths = [config[field]] if field == "first_frame" and field in config else config.get(field, [])
        for index, name in enumerate(paths):
            path = (Path(directory) / name).resolve()
            content = path.read_bytes()
            if not content:
                raise ValueError(f"Empty input: {path}")
            remote = f"{field}-{index}{path.suffix.lower()}"
            files[remote] = content
            names.append(remote)
        if field in config:
            packed[field] = names[0] if field == "first_frame" else names
    return packed, files


def environment(config, inputs):
    validate(config)
    # A base request must not inherit Turbo, LoRAs or another render's frame/reference settings.
    system = {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TMP", "TEMP", "LD_LIBRARY_PATH", "LD_PRELOAD",
              "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES",
              "NVIDIA_DRIVER_CAPABILITIES"}
    env = {key: value for key, value in os.environ.items() if key in system}
    mode = config["mode"]
    env.update(PREFIX="/opt/studio-runtime", SPARK_MODELS="/weights", SPARK_ENGINE="comfy", SPARK_BASE="h3",
               SPARK_SPARSITY="0", SPARK_COMFY_ARGS="--use-pytorch-cross-attention", REF_SIZE="max",
               REF_TURBO="1" if mode == "reference-turbo" else "0", ENGINE="fasth3",
               SPARK_VAE="minimax_h3_video_vae_fp16.safetensors" if mode == "first-frame"
               else "minimax_h3_video_vae_int8_convrot.safetensors",
               STEPS=str({"first-frame": 25, "reference": 20, "reference-turbo": 4}[mode]))
    for field in ("width", "height", "frames", "seed"):
        env[field.upper()] = str(config[field])
    if mode == "first-frame":
        env["FIRST_FRAME"] = str(Path(inputs) / config["first_frame"])
    for field, variable in (("images", "REF_IMAGES"), ("videos", "REF_VIDEOS"), ("audios", "REF_AUDIOS")):
        env[variable] = "\n".join(str(Path(inputs) / name) for name in config.get(field, []))
    return env


def inspect_output(path, config):
    probe = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)], text=True, timeout=30))
    video = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
    if not video or (video["width"], video["height"]) != (config["width"], config["height"]) or int(video.get("nb_frames", 0)) != config["frames"]:
        raise ValueError("Output dimensions or frame count differ from the requested clip")
    if not any(stream["codec_type"] == "audio" for stream in probe["streams"]):
        raise ValueError("Output has no audio stream")
    return probe


def verify(path, entry):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    if Path(path).stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"Weight verification failed: {entry['file']}")


def selected_models(manifest, mode):
    if mode not in MODES:
        raise ValueError("Unknown mode")
    return [entry for entry in manifest["files"] if mode in entry["modes"]]


def write_json(path, value):
    target = Path(path)
    part = target.with_suffix(".part")
    part.write_text(json.dumps(value, indent=2) + "\n")
    part.replace(target)
