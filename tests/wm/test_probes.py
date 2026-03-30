import pytest
import torch
from torch import nn

from stable_worldmodel.wm.probes import attach_probe, get_probe, load_probe


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


class SimpleProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(x)


######################
## attach_probe tests ##
######################


def test_attach_probe_creates_probes_dict():
    model = DummyModel()
    probe = SimpleProbe()
    attach_probe(model, 'my_probe', probe)
    assert hasattr(model, '_probes')
    assert 'my_probe' in model._probes


def test_attach_probe_stores_correct_module():
    model = DummyModel()
    probe = SimpleProbe()
    attach_probe(model, 'p', probe)
    assert model._probes['p'] is probe


def test_attach_probe_multiple_probes():
    model = DummyModel()
    p1 = SimpleProbe()
    p2 = SimpleProbe()
    attach_probe(model, 'a', p1)
    attach_probe(model, 'b', p2)
    assert 'a' in model._probes
    assert 'b' in model._probes


def test_attach_probe_requires_nn_module():
    model = DummyModel()
    with pytest.raises(AssertionError):
        attach_probe(model, 'bad', lambda x: x)


def test_attach_probe_overwrite():
    model = DummyModel()
    p1 = SimpleProbe()
    p2 = SimpleProbe()
    attach_probe(model, 'p', p1)
    attach_probe(model, 'p', p2)
    assert model._probes['p'] is p2


####################
## get_probe tests ##
####################


def test_get_probe_returns_none_without_probes():
    model = DummyModel()
    assert get_probe(model, 'missing') is None


def test_get_probe_returns_none_for_missing_key():
    model = DummyModel()
    probe = SimpleProbe()
    attach_probe(model, 'a', probe)
    assert get_probe(model, 'b') is None


def test_get_probe_returns_attached_probe():
    model = DummyModel()
    probe = SimpleProbe()
    attach_probe(model, 'p', probe)
    assert get_probe(model, 'p') is probe


#####################
## load_probe tests ##
#####################


def test_load_probe_from_module(tmp_path):
    model = DummyModel()
    probe = SimpleProbe()
    path = tmp_path / 'probe.pt'
    torch.save(probe, path)

    load_probe(model, 'p', path)

    loaded = get_probe(model, 'p')
    assert loaded is not None
    assert isinstance(loaded, SimpleProbe)


def test_load_probe_from_state_dict(tmp_path):
    model = DummyModel()
    probe = SimpleProbe()
    attach_probe(model, 'p', probe)

    path = tmp_path / 'state.pt'
    torch.save(probe.state_dict(), path)

    load_probe(model, 'p', path)

    loaded = get_probe(model, 'p')
    assert loaded is not None


def test_load_probe_state_dict_no_probe_raises(tmp_path):
    model = DummyModel()
    probe = SimpleProbe()
    path = tmp_path / 'state.pt'
    torch.save(probe.state_dict(), path)

    with pytest.raises(ValueError, match='No probe found'):
        load_probe(model, 'missing', path)


def test_load_probe_state_dict_updates_weights(tmp_path):
    model = DummyModel()
    probe = SimpleProbe()
    attach_probe(model, 'p', probe)

    # Save modified weights
    new_probe = SimpleProbe()
    with torch.no_grad():
        new_probe.fc.weight.fill_(99.0)
    path = tmp_path / 'state.pt'
    torch.save(new_probe.state_dict(), path)

    load_probe(model, 'p', path)

    assert torch.allclose(get_probe(model, 'p').fc.weight, torch.full_like(probe.fc.weight, 99.0))
