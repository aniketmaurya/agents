import json

from agents.tools import get_current_weather


def test_get_current_weather():
    current_weather = json.loads(get_current_weather("San Francisco"))
    assert isinstance(current_weather, dict)
    assert "FeelsLikeC" in current_weather
