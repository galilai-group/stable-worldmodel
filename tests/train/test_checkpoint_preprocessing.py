"""Tests for training checkpoint preprocessing wiring."""

import importlib
from unittest.mock import patch

import pytest
import torch


@pytest.mark.parametrize(
    'module_name',
    [
        'scripts.train.lewm',
        'scripts.train.pldm',
        'scripts.train.prejepa',
    ],
)
def test_save_callback_passes_fitted_preprocessing(module_name):
    module = importlib.import_module(module_name)

    preprocessing = {'action': object(), 'state': object()}
    callback = module.SaveCkptCallback(
        run_name='test-run',
        cfg={'_target_': 'dummy.Model'},
        preprocessing=preprocessing,
        epoch_interval=1,
    )
    model = torch.nn.Linear(2, 2)

    with patch.object(module, 'save_pretrained') as save_mock:
        callback._save(model, epoch=3)

    save_mock.assert_called_once()
    kwargs = save_mock.call_args.kwargs

    assert kwargs['run_name'] == 'test-run'
    assert kwargs['filename'] == 'weights_epoch_3.pt'
    assert kwargs['preprocessing'] is preprocessing
