# H3 on Modal

Render from a first frame or from image/video/audio references using Imagination Studio by Keys (drowzeys) and
Sam Wasserman / Wasserman Productions. The batch runner uses the existing guarded ComfyUI scripts. It does not
deploy a web service or require a local GPU, Docker, the desktop app, or the full model installer.

## Setup

Clone this fork and check out the PR branch (`codex/modal-h3`, or `main` after merge). From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r modal/requirements.txt
modal setup
modal profile list
export MODAL_PROFILE=your-profile
```

Use the profile connected to the workspace that should pay for the run. The runner requires an explicit
`MODAL_PROFILE`; use that same profile and Modal environment for preparation, rendering and collection.
Teammates can each use their own workspace, or use profiles connected to the same shared workspace. Volumes
belong to that workspace/environment; personal volumes are not automatically accessible from another one.
Do not share API tokens or put credentials in JSON configs. Modal authenticates the local CLI; no credentials
or access keys are passed to the rendering subprocess.

The runner creates `studio-h3-models` and `studio-h3-outputs` in the selected environment. Preparation uses CPU
compute. Rendering requests one B200, falling back to an H200, with 256 GiB host RAM, at most one GPU container
per app, no automatic retries, and a 24-hour ceiling. Modal compute and storage charges apply. Concurrent
preparation and rendering against the same model volume are unsupported; prepare first, then render.

Models are downloaded directly from Comfy Org's MiniMax H3 repository at revision
`912b6e38b2092ec4d2e86d24183e0d59ea059d48`. Preparation checks every required file's size and SHA-256 against
`models.json`, including files already present. A mismatched file fails preparation; remove that file from the
model volume and prepare again. Render startup requires the matching preparation marker and checks file sizes.
No model weights are included in Git. MiniMax H3's Community License and territory restrictions apply; consult
the model's licence before commercial use. Studio's Apache licence does not replace the weights' licence.

## First-frame video

Copy your approved portrait to `modal/examples/portrait.png`, then edit the prompt in
`modal/examples/first-frame.json`. Media paths are relative to the JSON file. The example requests 124 frames
at 24 fps (5.167 seconds), 768 × 1344 pixels, seed 195. The mode fixes base FL2VA at 25 passes with the fp16 video
VAE. This preserves the recipe of the first-frame experiments; it does not guarantee face identity or realism.

```bash
modal run modal/h3.py --phase prepare --mode first-frame
modal run --detach modal/h3.py --phase render --config modal/examples/first-frame.json --output outputs/my-first-frame
```

Preparation downloads about 68.2 GiB for this mode. Shared files are reused across modes. Use `--detach` for
rendering so the Modal app can continue after the local client disconnects. Keep the local output directory:
`submission.json` records the call ID and run ID before waiting for the result. The runner downloads `video.mp4`,
`receipt.json`, `render.log` and `comfy.log` there when complete.

To resume waiting and download the result without resubmitting:

```bash
modal run modal/h3.py --phase collect --output outputs/my-first-frame
```

A directory with an existing `submission.json` cannot submit again. If it says `submitting` without a call ID,
the outcome is uncertain: inspect the app's calls in the Modal dashboard and the run ID in `studio-h3-outputs`
before deciding whether another submission is needed. Do not delete a receipt to retry an uncertain call.
`collect` also depends on Modal retaining the call result; persisted output files can be retrieved independently:

```bash
modal volume get studio-h3-outputs RUN_ID/video.mp4 outputs/recovered.mp4
modal volume get studio-h3-outputs RUN_ID/render.log outputs/render.log
modal volume get studio-h3-outputs RUN_ID/receipt.json outputs/receipt.json
```

The first-frame mode generates a voice. It cannot combine the first frame with a voice reference.

## Reference video

Place `portrait.png` and a clean `voice.wav` beside `modal/examples/reference-video.json`. Use references you
have permission to use. The example uses the portrait as `<Picture 1>` and the audio as `<Audio 1>`. Audio is
optional; remove `audios` from the config for an image-only reference render.

```bash
modal run modal/h3.py --phase prepare --mode reference
modal run --detach modal/h3.py --phase render --config modal/examples/reference-video.json --output outputs/my-reference
```

Reference mode uses base Ref2VA at 20 passes and the int8 video VAE (about 65.9 GiB of weights). To use the
four-pass adapter, change the config's `mode` to `reference-turbo`, prepare that mode, and choose a fresh output
directory. The adapter adds about 1.82 GiB. Each mode clears inherited Turbo, LoRA and conditioning variables.
Reference guidance is fixed at the renderer's default; the runner does not expose a CFG control.

`images`, `videos` and `audios` are separate lists: up to nine images, three videos, three audio clips, and
twelve files total. At least one image or video is required. Video/audio reference duration must be 2–15 seconds
per file and at most 15 seconds total per kind; Studio probes and rejects violations before loading the model.
An image reference is not a first frame. Voice references are conditioning, not a promise of voice cloning.

Both modes accept `prompt`, `width`, `height`, `frames` and `seed`. Dimensions must be multiples of 32 between
256 and 1536, with area at most 768 × 1344 pixels. Frames must be `17n + 5`, from 22 to 362. There is no negative
prompt field. The receipt records exact mode, pass count, dimensions, frames, seed, model revision, decoder and
device. The runner verifies the output frame count, dimensions and presence of audio; speech accuracy, identity,
pronunciation, lip sync and aesthetic quality require listening and visual review.

## Persistence and cleanup

Each run has a separate directory in `studio-h3-outputs`, including its input copies, receipts and logs. After
each job the guarded ComfyUI process stops and the single-use GPU container exits. The volumes remain, so weights
are reused. Retrieve failures' receipts and logs with `modal volume get`; failed calls are not resubmitted.
To cancel active compute, find the specific `studio-h3-batch` app run in the Modal dashboard and stop it. Inspect
any incomplete output before another paid run.

After downloading everything needed, remove a run directory or delete the whole volume only when no run is
active and nobody sharing the workspace needs it. `modal volume rm studio-h3-outputs RUN_ID --recursive` removes
that run, including uploaded references. `modal volume delete studio-h3-models` deletes all cached weights and
requires downloading them again. These are manual cleanup commands; the runner never deletes shared weights.

## Validation boundary

Run `python -m pytest -q tests/test_modal_h3.py tests/test_memory_guard.py` for local config, file, checksum,
environment and memory-guard checks. CI also runs these with the full existing suite and imports the Modal CLI.
These checks do not emulate Modal, a GPU or successful generation. The runtime recipe derives from actual
Modal H3 experiments; a fresh workspace prepare/render/collect run of this packaged runner still needs live
qualification before declaring it supported for unattended team use.

## Authenticated HTTP service

`modal/service.py` adds the restricted upload, video submission, polling and completed-file routes used by
VibeGrow. It does not serve the Studio UI. Every route requires `X-Studio-Key`; ComfyUI stays private.
Job status and call IDs live in a Modal Dict and files in the output Volume, so restarting the web process
keeps accepted work. Ambiguous submissions are never automatically repeated. Rendering uses one GPU container,
no automatic retries and the batch runner's guarded modes and output checks.

1. Set `MODAL_PROFILE` explicitly. Prepare the selected mode with the batch command above.
2. Create a private dotenv file containing `STUDIO_ACCESS_KEY` with a random value of at least 32 characters.
   Run `python -m modal secret create studio-h3-service --from-dotenv /secure/path/studio.env`.
3. Run `python -m modal deploy modal/service.py`. Save the printed HTTPS endpoint.
4. Check `/health` with the key, then qualify a real render, download and web-process restart before connecting clients.
   Configure VibeGrow's server-only `STUDIO_CONNECTION` with `url`, `key` and only the qualified `models`.

The batch runner and service share `studio-h3-models` and `studio-h3-outputs` by default. Set
`STUDIO_MODELS_VOLUME` and `STUDIO_OUTPUTS_VOLUME` consistently during preparation and deployment to reuse an
existing installation. Preserve those names when redeploying. Keep outputs and job records until clients have
reconciled and retained their results; storage cleanup is manual. Modal compute and storage charges belong to
the selected workspace. See the root licence notices before commercial use; weights are not redistributed.
