"""Tests for stable_worldmodel.wm.utils (save_pretrained / load_pretrained)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from stable_worldmodel.wm.utils import (
    _load_config,
    _resolve,
    _resolve_folder,
    load_pretrained,
    save_pretrained,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    def __init__(self, in_features: int = 4, out_features: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


TINY_CONFIG = {
    '_target_': 'tests.wm.test_utils.TinyModel',
    'in_features': 4,
    'out_features': 2,
}
# save_pretrained requires an OmegaConf DictConfig (it calls OmegaConf.to_container internally)
TINY_OMEGACONF = OmegaConf.create(TINY_CONFIG)


def _ckpt_root(tmp_path: Path) -> Path:
    return tmp_path / 'checkpoints'


def _make_checkpoint(tmp_path: Path, run_name: str, model: nn.Module) -> Path:
    """Write a minimal checkpoint (weights.pt + config.json) and return the run dir."""
    run_dir = _ckpt_root(tmp_path) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / 'weights.pt')
    (run_dir / 'config.json').write_text(json.dumps(TINY_CONFIG))
    return run_dir


# ---------------------------------------------------------------------------
# save_pretrained
# ---------------------------------------------------------------------------


def test_save_pretrained_saves_weights_file(tmp_path):
    model = TinyModel()
    save_pretrained(
        model, run_name='run1', config=TINY_OMEGACONF, cache_dir=tmp_path
    )
    assert (_ckpt_root(tmp_path) / 'run1' / 'weights.pt').exists()


def test_save_pretrained_saves_config_json(tmp_path):
    model = TinyModel()
    save_pretrained(
        model, run_name='run1', config=TINY_OMEGACONF, cache_dir=tmp_path
    )
    saved = json.loads(
        (_ckpt_root(tmp_path) / 'run1' / 'config.json').read_text()
    )
    assert saved['_target_'] == TINY_CONFIG['_target_']


def test_save_pretrained_state_dict_matches(tmp_path):
    model = TinyModel()
    save_pretrained(
        model, run_name='run1', config=TINY_OMEGACONF, cache_dir=tmp_path
    )
    loaded = torch.load(
        _ckpt_root(tmp_path) / 'run1' / 'weights.pt', map_location='cpu'
    )
    for key in model.state_dict():
        torch.testing.assert_close(model.state_dict()[key], loaded[key])


def test_save_pretrained_custom_filename(tmp_path):
    model = TinyModel()
    save_pretrained(
        model,
        run_name='run1',
        config=TINY_OMEGACONF,
        filename='epoch_5.pt',
        cache_dir=tmp_path,
    )
    assert (_ckpt_root(tmp_path) / 'run1' / 'epoch_5.pt').exists()


def test_save_pretrained_no_config_skips_json(tmp_path):
    model = TinyModel()
    save_pretrained(model, run_name='run1', cache_dir=tmp_path)
    assert (_ckpt_root(tmp_path) / 'run1' / 'weights.pt').exists()
    assert not (_ckpt_root(tmp_path) / 'run1' / 'config.json').exists()


def test_save_pretrained_config_key_extracts_subconfig(tmp_path):
    model = TinyModel()
    cfg = OmegaConf.create({'model': TINY_CONFIG, 'training': {'lr': 1e-3}})
    save_pretrained(
        model,
        run_name='run1',
        config=cfg,
        config_key='model',
        cache_dir=tmp_path,
    )
    saved = json.loads(
        (_ckpt_root(tmp_path) / 'run1' / 'config.json').read_text()
    )
    assert '_target_' in saved
    assert 'training' not in saved


def test_save_pretrained_creates_run_directory(tmp_path):
    model = TinyModel()
    save_pretrained(
        model,
        run_name='new/nested/run',
        config=TINY_OMEGACONF,
        cache_dir=tmp_path,
    )
    assert (
        _ckpt_root(tmp_path) / 'new' / 'nested' / 'run' / 'weights.pt'
    ).exists()


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------


def test_load_config_returns_dict(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps(TINY_CONFIG))
    result = _load_config(tmp_path)
    assert isinstance(result, dict)
    assert result['_target_'] == TINY_CONFIG['_target_']


def test_load_config_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match='config.json not found'):
        _load_config(tmp_path)


# ---------------------------------------------------------------------------
# _resolve_folder
# ---------------------------------------------------------------------------


def test_resolve_folder_returns_pt_and_config(tmp_path):
    torch.save(TinyModel().state_dict(), tmp_path / 'weights.pt')
    (tmp_path / 'config.json').write_text(json.dumps(TINY_CONFIG))
    pt_path, config = _resolve_folder(tmp_path)
    assert pt_path == tmp_path / 'weights.pt'
    assert config['_target_'] == TINY_CONFIG['_target_']


def test_resolve_folder_raises_when_no_pt(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps(TINY_CONFIG))
    with pytest.raises(FileNotFoundError, match='No .pt file found'):
        _resolve_folder(tmp_path)


def test_resolve_folder_raises_when_multiple_pt(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps(TINY_CONFIG))
    torch.save(TinyModel().state_dict(), tmp_path / 'a.pt')
    torch.save(TinyModel().state_dict(), tmp_path / 'b.pt')
    with pytest.raises(ValueError, match='Ambiguous checkpoint'):
        _resolve_folder(tmp_path)


# ---------------------------------------------------------------------------
# _resolve
# ---------------------------------------------------------------------------


def test_resolve_explicit_pt_file(tmp_path):
    run_dir = tmp_path / 'run1'
    run_dir.mkdir()
    torch.save(TinyModel().state_dict(), run_dir / 'weights.pt')
    (run_dir / 'config.json').write_text(json.dumps(TINY_CONFIG))
    pt_path, _ = _resolve('run1/weights.pt', tmp_path)
    assert pt_path == run_dir / 'weights.pt'


def test_resolve_explicit_pt_not_found_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match='Checkpoint not found'):
        _resolve('nonexistent/weights.pt', tmp_path)


def test_resolve_folder_format(tmp_path):
    run_dir = tmp_path / 'run1'
    run_dir.mkdir()
    torch.save(TinyModel().state_dict(), run_dir / 'weights.pt')
    (run_dir / 'config.json').write_text(json.dumps(TINY_CONFIG))
    pt_path, _ = _resolve('run1', tmp_path)
    assert pt_path.suffix == '.pt'


def test_resolve_invalid_name_raises(tmp_path):
    with pytest.raises(ValueError, match='Cannot resolve'):
        _resolve('this_does_not_exist', tmp_path)


def test_resolve_hf_repo_triggers_download(tmp_path):
    with patch('stable_worldmodel.wm.utils._resolve_hf') as mock_hf:
        mock_hf.return_value = (tmp_path / 'weights.pt', TINY_CONFIG)
        _resolve('some-user/some-repo', tmp_path)
        mock_hf.assert_called_once_with('some-user/some-repo', tmp_path)


# ---------------------------------------------------------------------------
# load_pretrained
# ---------------------------------------------------------------------------


def test_load_pretrained_from_explicit_pt(tmp_path):
    _make_checkpoint(tmp_path, 'run1', TinyModel())
    with patch('hydra.utils.instantiate', return_value=TinyModel()):
        loaded = load_pretrained('run1/weights.pt', cache_dir=tmp_path)
    assert isinstance(loaded, TinyModel)


def test_load_pretrained_from_folder(tmp_path):
    _make_checkpoint(tmp_path, 'run1', TinyModel())
    with patch('hydra.utils.instantiate', return_value=TinyModel()):
        loaded = load_pretrained('run1', cache_dir=tmp_path)
    assert isinstance(loaded, TinyModel)


def test_load_pretrained_weights_match(tmp_path):
    original = TinyModel()
    _make_checkpoint(tmp_path, 'run1', original)
    with patch('hydra.utils.instantiate', return_value=TinyModel()):
        loaded = load_pretrained('run1/weights.pt', cache_dir=tmp_path)
    for key in original.state_dict():
        torch.testing.assert_close(
            original.state_dict()[key], loaded.state_dict()[key]
        )


def test_load_pretrained_instantiate_called_with_config(tmp_path):
    _make_checkpoint(tmp_path, 'run1', TinyModel())
    with patch('hydra.utils.instantiate') as mock_inst:
        mock_inst.return_value = TinyModel()
        load_pretrained('run1/weights.pt', cache_dir=tmp_path)
    assert mock_inst.call_args[0][0]['_target_'] == TINY_CONFIG['_target_']


def test_load_pretrained_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pretrained('ghost/weights.pt', cache_dir=tmp_path)


def test_load_pretrained_missing_config_raises(tmp_path):
    run_dir = _ckpt_root(tmp_path) / 'run_no_cfg'
    run_dir.mkdir(parents=True)
    torch.save(TinyModel().state_dict(), run_dir / 'weights.pt')
    with patch('hydra.utils.instantiate', return_value=TinyModel()):
        with pytest.raises(FileNotFoundError, match='config.json not found'):
            load_pretrained('run_no_cfg', cache_dir=tmp_path)


def test_load_pretrained_uses_hf_cache_without_downloading(tmp_path):
    repo_id = 'myuser/myrepo'
    cache_name = f'models--{repo_id.replace("/", "--")}'
    _make_checkpoint(tmp_path, cache_name, TinyModel())
    with patch('hydra.utils.instantiate', return_value=TinyModel()):
        with patch('stable_worldmodel.wm.utils._download') as mock_dl:
            loaded = load_pretrained(repo_id, cache_dir=tmp_path)
            mock_dl.assert_not_called()
    assert isinstance(loaded, TinyModel)


# ---------------------------------------------------------------------------
# Round-trip integration
# ---------------------------------------------------------------------------


def test_roundtrip_preserves_weights(tmp_path):
    original = TinyModel()
    save_pretrained(
        original, run_name='rt_run', config=TINY_OMEGACONF, cache_dir=tmp_path
    )
    with patch('hydra.utils.instantiate', return_value=TinyModel()):
        loaded = load_pretrained('rt_run', cache_dir=tmp_path)
    for key in original.state_dict():
        torch.testing.assert_close(
            original.state_dict()[key], loaded.state_dict()[key]
        )


def test_roundtrip_custom_filename(tmp_path):
    original = TinyModel()
    save_pretrained(
        original,
        run_name='rt_run',
        config=TINY_OMEGACONF,
        filename='epoch_3.pt',
        cache_dir=tmp_path,
    )
    with patch('hydra.utils.instantiate', return_value=TinyModel()):
        loaded = load_pretrained('rt_run/epoch_3.pt', cache_dir=tmp_path)
    for key in original.state_dict():
        torch.testing.assert_close(
            original.state_dict()[key], loaded.state_dict()[key]
        )
