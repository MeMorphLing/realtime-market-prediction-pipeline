"""Smoke tests for model construction and forward pass."""

import torch

from src.models import build_gru, build_lstm, build_rnn


def _check_forward(model, input_size: int) -> None:
    x = torch.zeros(4, 10, input_size, dtype=torch.float32)
    out = model(x)
    assert out.shape == (4, 1), f"expected (4, 1), got {tuple(out.shape)}"


def test_rnn_forward() -> None:
    _check_forward(build_rnn(input_size=9), input_size=9)


def test_lstm_forward() -> None:
    _check_forward(build_lstm(input_size=9), input_size=9)


def test_gru_forward() -> None:
    _check_forward(build_gru(input_size=9), input_size=9)


def test_models_produce_different_outputs() -> None:
    """Different model classes shouldn't collapse to the same weights/output."""
    torch.manual_seed(0)
    x = torch.randn(2, 10, 9)
    a = build_rnn(input_size=9)(x).detach()
    b = build_lstm(input_size=9)(x).detach()
    c = build_gru(input_size=9)(x).detach()
    assert not torch.allclose(a, b)
    assert not torch.allclose(b, c)
