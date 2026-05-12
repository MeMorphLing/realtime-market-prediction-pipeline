"""Smoke tests for VADER sentiment scoring."""

from src.sentiment.vader import classify_vader


def test_vader_positive() -> None:
    out = classify_vader(["Stocks soared on strong earnings."])
    assert len(out) == 1
    assert out[0]["label"] == "positive"
    assert out[0]["compound"] > 0.05


def test_vader_negative() -> None:
    out = classify_vader(["Markets crashed amid recession fears."])
    assert out[0]["label"] == "negative"
    assert out[0]["compound"] < -0.05


def test_vader_neutral() -> None:
    out = classify_vader(["The Fed meets on Wednesday."])
    assert out[0]["label"] == "neutral"


def test_vader_empty_input() -> None:
    assert classify_vader([]) == []
