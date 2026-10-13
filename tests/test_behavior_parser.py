"""Tests for natural language cloth behavior parser."""

import pytest
from autonomous_cfx.agent.behavior_parser import BehaviorParser


def test_parse_leather_prompt():
    prompt = "heavy leather jacket, stiff collar, low stretch, high friction with body, high damping"
    res = BehaviorParser.parse(prompt)

    assert res.detected_material == "leather"
    assert "High Density" in res.detected_traits
    assert "High Stretch Resistance" in res.detected_traits
    assert "High Bend Stiffness" in res.detected_traits
    assert "High Friction" in res.detected_traits
    assert res.properties.stretch_compliance < 1e-5
    assert res.properties.friction >= 0.55


def test_parse_silk_prompt():
    prompt = "lightweight silk windbreaker, loose drape, soft folds, low friction"
    res = BehaviorParser.parse(prompt)

    assert res.detected_material == "silk"
    assert "Low Density" in res.detected_traits
    assert "Soft Bend Drape" in res.detected_traits
    assert "Low Friction" in res.detected_traits
    assert res.properties.density < 0.2
    assert res.properties.bend_compliance > 0.01


def test_parse_denim_prompt():
    prompt = "rugged denim jacket, tight fit, high stretch resistance"
    res = BehaviorParser.parse(prompt)

    assert res.detected_material == "denim"
    assert "High Stretch Resistance" in res.detected_traits
