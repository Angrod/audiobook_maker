import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from view import find_matching_preset  # noqa: E402

with open(os.path.join(ROOT, "configs", "s2s_config.json"), encoding="utf-8") as f:
    RVC = next(e for e in json.load(f)["s2s_engines"] if e["name"] == "RVC")
PARAMS = {p["attribute"]: p for p in RVC["parameters"]}
PRESETS = PARAMS["rvc_preset"]["options"]


def test_presets_only_set_existing_controls_with_valid_values():
    for name, values in PRESETS.items():
        for attribute, value in values.items():
            param = PARAMS[attribute]
            assert param.get("advanced"), f"{name}: {attribute} should live under Advanced settings"
            if param["type"] == "slider":
                assert param["min"] <= value <= param["max"], (name, attribute)
            else:
                assert value in param["custom_options"], (name, attribute)


def test_recommended_preset_matches_the_defaults():
    recommended = next(iter(PRESETS))
    assert "recommended" in recommended
    defaults = {a: p.get("default", (p.get("custom_options") or [None])[0]) for a, p in PARAMS.items()}
    assert find_matching_preset(PRESETS, defaults) == recommended


def test_custom_when_values_do_not_match_any_preset():
    values = dict(PRESETS["Closer to target voice"], filter_radius=6)
    assert find_matching_preset(PRESETS, values) == "Custom"
    assert find_matching_preset(PRESETS, dict(PRESETS["Fewer artifacts"])) == "Fewer artifacts"


def test_everyday_controls_are_not_hidden():
    assert not PARAMS["selected_voice"].get("advanced")
    assert not PARAMS["rvc_preset"].get("advanced")
    assert not PARAMS["f0pitch"].get("advanced")
