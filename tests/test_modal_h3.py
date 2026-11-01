"""Pure input/file/environment checks; these do not stand in for a Modal GPU render."""
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h3_support", ROOT / "modal/h3_support.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def first_frame():
    return dict(mode="first-frame", prompt="A person speaks.", width=768, height=1344,
                frames=124, seed=195, first_frame="portrait.png")


@pytest.mark.parametrize("change", [
    {"audios": ["voice.wav"]}, {"images": ["person.png"]}, {"frames": 120}, {"width": 769},
    {"width": 1536, "height": 1536}, {"seed": True}, {"seed": -1}, {"steps": 4}, {"prompt": " "},
    {"mode": "reference"}, {"first_frame": ""}, {"frames": 379},
])
def test_rejects_invalid_or_mixed_render_modes(change):
    with pytest.raises(ValueError):
        H.validate(first_frame() | change)


def test_reference_inputs_require_visual_media_and_obey_limits():
    config = first_frame()
    del config["first_frame"]
    config.update(mode="reference", images=["person.png"], audios=["voice.wav"])
    assert H.validate(config) == config
    for change in ({"images": []}, {"videos": ["v.mp4"] * 4}, {"images": ["p.png"] * 9, "audios": ["a.wav"] * 3,
                                                "videos": ["v.mp4"]}):
        with pytest.raises(ValueError):
            H.validate(config | change)


def test_packs_real_files_relative_to_config_without_sending_client_paths(tmp_path):
    (tmp_path / "portrait with spaces.png").write_bytes(b"image input")
    packed, files = H.pack_inputs(first_frame() | {"first_frame": "portrait with spaces.png"}, tmp_path)
    assert packed["first_frame"] == "first_frame-0.png"
    assert files == {"first_frame-0.png": b"image input"}
    (tmp_path / "portrait with spaces.png").write_bytes(b"")
    with pytest.raises(ValueError, match="Empty input"):
        H.pack_inputs(first_frame() | {"first_frame": "portrait with spaces.png"}, tmp_path)


def test_mode_environment_clears_inherited_turbo_and_other_conditioning(monkeypatch):
    for key, value in {"REF_TURBO": "1", "LAST_FRAME": "old.png", "LORA": "old.safetensors",
                       "SPARK_REF_CFG": "5", "REF_VIDEO_SOUND": "1", "PROMPT_FILE": "old.txt",
                       "MODAL_TOKEN_SECRET": "test-placeholder"}.items():
        monkeypatch.setenv(key, value)
    env = H.environment(first_frame(), "/inputs")
    assert env["REF_TURBO"] == "0"
    assert env["STEPS"] == "25"
    assert env["FIRST_FRAME"] == "/inputs/portrait.png"
    assert env["SPARK_VAE"] == "minimax_h3_video_vae_fp16.safetensors"
    assert env["REF_AUDIOS"] == env["REF_IMAGES"] == ""
    assert not ({"LAST_FRAME", "LORA", "SPARK_REF_CFG", "REF_VIDEO_SOUND", "PROMPT_FILE", "MODAL_TOKEN_SECRET"} & env.keys())
    config = first_frame()
    del config["first_frame"]
    config.update(mode="reference-turbo", images=["portrait.png"])
    env = H.environment(config, "/inputs")
    assert (env["REF_TURBO"], env["STEPS"]) == ("1", "4")
    assert env["REF_IMAGES"] == "/inputs/portrait.png"
    assert "FIRST_FRAME" not in env
    config["mode"] = "reference"
    assert (H.environment(config, "/inputs")["REF_TURBO"], H.environment(config, "/inputs")["STEPS"]) == ("0", "20")


def test_checksum_verification_rejects_same_size_corruption_and_truncation(tmp_path):
    path = tmp_path / "model"
    content = b"model data"
    entry = {"file": "model", "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    path.write_bytes(content)
    H.verify(path, entry)
    for changed in (b"MODEL DATA", b"model"):
        path.write_bytes(changed)
        with pytest.raises(ValueError, match="Weight verification failed"):
            H.verify(path, entry)


def test_manifest_selects_only_required_weights_and_pins_the_tested_revision():
    manifest = json.loads((ROOT / "modal/models.json").read_text())
    assert manifest["revision"] == H.MODEL_REVISION
    first = {x["file"] for x in H.selected_models(manifest, "first-frame")}
    base = {x["file"] for x in H.selected_models(manifest, "reference")}
    turbo = {x["file"] for x in H.selected_models(manifest, "reference-turbo")}
    assert "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors" in first
    assert "diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors" in base
    assert not any("loras/" in x or "ref2va" in x for x in first)
    assert len(turbo - base) == 1 and next(iter(turbo - base)).startswith("loras/")
    assert len(first & base) == 2


def test_json_receipt_replacement_leaves_a_complete_document(tmp_path):
    path = tmp_path / "receipt.json"
    H.write_json(path, {"state": "submitted"})
    H.write_json(path, {"state": "completed", "outputs": ["video.mp4"]})
    assert json.loads(path.read_text()) == {"state": "completed", "outputs": ["video.mp4"]}
    assert not path.with_suffix(".part").exists()


def test_output_inspection_reads_real_media_and_rejects_wrong_geometry_frames_and_missing_audio(tmp_path):
    video = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=256x256:r=24",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000", "-t", "0.916667",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)], check=True, timeout=30)
    config = first_frame() | {"width": 256, "height": 256, "frames": 22}
    probe = H.inspect_output(video, config)
    assert {stream["codec_type"] for stream in probe["streams"]} == {"video", "audio"}
    for change in ({"width": 512}, {"frames": 39}):
        with pytest.raises(ValueError, match="dimensions or frame count"):
            H.inspect_output(video, config | change)
    silent = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-c:v", "copy", "-an", str(silent)],
                   check=True, timeout=30)
    with pytest.raises(ValueError, match="no audio"):
        H.inspect_output(silent, config)
