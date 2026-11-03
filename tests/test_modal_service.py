"""Pure HTTP-contract validation; no claim of Modal or GPU execution."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'modal'))
from service_support import video_config


def request(**params):
    return {'kind': 'video', 'params': {'prompt': 'A person speaks.', 'width': 768, 'height': 1344,
            'seconds': 5, 'seed': 42, 'image': 'uploads/abc.png', 'preset': 'custom', 'ref_turbo': False,
            'steps': 25, **params}}


def test_first_frame_preserves_dimensions_and_identity_input():
    config = video_config(request())
    assert config == {'mode': 'first-frame', 'prompt': 'A person speaks.', 'width': 768, 'height': 1344,
                      'frames': 124, 'seed': 42, 'first_frame': 'abc.png', 'images': [], 'videos': [], 'audios': []}


@pytest.mark.parametrize('change', [
    {'image': '../portrait.png'}, {'image': 'uploads/a.mp4'}, {'width': 769}, {'height': 1536},
    {'seconds': float('nan')}, {'seconds': True}, {'seconds': 16}, {'seed': True}, {'steps': 4},
    {'ref_turbo': True}, {'references': ['uploads/a.png']}, {'reference_audios': ['uploads/a.wav']},
    {'output': '/tmp/out.mp4'}, {'preset': 'anything'}, {'reference_videos': 'uploads/a.mp4'}])
def test_rejects_unsafe_or_mixed_parameters(change):
    with pytest.raises(ValueError):
        video_config(request(**change))


@pytest.mark.parametrize('turbo,steps,mode', [(False, 20, 'reference'), (True, 4, 'reference-turbo')])
def test_reference_modes_and_audio_keep_distinct_roles(turbo, steps, mode):
    body = request(references=['uploads/a.png'], reference_audios=['uploads/b.wav'], ref_turbo=turbo, steps=steps)
    del body['params']['image']
    config = video_config(body)
    assert config['mode'] == mode
    assert config['images'] == ['a.png']
    assert config['audios'] == ['b.wav']
    with pytest.raises(ValueError):
        body['params']['references'] = ['uploads/b.wav']
        video_config(body)
