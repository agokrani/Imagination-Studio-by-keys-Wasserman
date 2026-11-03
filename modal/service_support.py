"""Translate the restricted Studio HTTP video contract to the guarded runner."""
import math
import re

from h3_support import validate

MEDIA = re.compile(r"uploads/([a-z0-9]+\.(png|jpg|jpeg|webp|mp4|mov|wav|mp3|m4a|ogg|flac))\Z")


def start_render(job, run_id):
    if job.get("run_id") != run_id:
        raise ValueError("Render run does not match the accepted job")
    if job["status"] == "queued":
        return {**job, "status": "running"}
    if job["status"] == "running":
        return {**job, "status": "unknown", "error": "Interrupted render requires operator reconciliation"}
    return job


def video_config(body):
    if not isinstance(body, dict) or set(body) - {"kind", "title", "params"} or body.get("kind") != "video":
        raise ValueError("Only a single video job is supported")
    p = body.get("params")
    allowed = {"prompt", "width", "height", "seconds", "seed", "image", "references", "reference_videos",
               "reference_audios", "preset", "ref_turbo", "steps"}
    if not isinstance(p, dict) or set(p) - allowed:
        raise ValueError("Unsupported video parameters")
    seconds = p.get("seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 2 <= seconds <= 15:
        raise ValueError("Duration must be 2–15 seconds")
    if p.get("preset") != "custom" or type(p.get("ref_turbo")) is not bool:
        raise ValueError("Explicit custom preset and Turbo selection required")
    first = "image" in p
    mode = "first-frame" if first else "reference-turbo" if p["ref_turbo"] else "reference"
    if (first and p["ref_turbo"]) or p.get("steps") != {"first-frame": 25, "reference": 20, "reference-turbo": 4}[mode]:
        raise ValueError("Invalid steps for the selected mode")
    config = {"mode": mode, "prompt": p.get("prompt"), "width": p.get("width"), "height": p.get("height"),
              "frames": 17 * round((seconds * 24 - 5) / 17) + 5, "seed": p.get("seed", 42)}
    if first:
        config["first_frame"] = media_name(p["image"], {"png", "jpg", "jpeg", "webp"})
    for source, target, extensions in (("references", "images", {"png", "jpg", "jpeg", "webp"}),
                                       ("reference_videos", "videos", {"mp4", "mov"}),
                                       ("reference_audios", "audios", {"wav", "mp3", "m4a", "ogg", "flac"})):
        items = p.get(source, [])
        if not isinstance(items, list):
            raise ValueError("References must be lists")
        config[target] = [media_name(item, extensions) for item in items]
    return validate(config)


def media_name(value, extensions):
    match = MEDIA.fullmatch(value) if isinstance(value, str) else None
    if not match or match[2] not in extensions:
        raise ValueError("Invalid owned media path or kind")
    return match[1]
