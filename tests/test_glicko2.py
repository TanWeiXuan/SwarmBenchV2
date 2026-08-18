import pytest

from swarmbench.competition.glicko2 import GlickoRating, simultaneous_update, update_rating


def test_reference_glicko2_example() -> None:
    player = GlickoRating(1500.0, 200.0, 0.06)
    results = [
        (GlickoRating(1400.0, 30.0, 0.06), 1.0),
        (GlickoRating(1550.0, 100.0, 0.06), 0.0),
        (GlickoRating(1700.0, 300.0, 0.06), 0.0),
    ]
    updated = update_rating(player, results)
    assert updated.rating == pytest.approx(1464.06, abs=0.02)
    assert updated.deviation == pytest.approx(151.52, abs=0.02)
    assert updated.volatility == pytest.approx(0.059996, abs=0.000002)


def test_draw_between_equal_players_keeps_rating() -> None:
    updated = update_rating(GlickoRating(), [(GlickoRating(), 0.5)])
    assert updated.rating == pytest.approx(1500.0)
    assert updated.deviation < 350.0


def test_inactivity_increases_deviation_only() -> None:
    original = GlickoRating(1600.0, 80.0, 0.06)
    updated = update_rating(original, [])
    assert updated.rating == original.rating
    assert updated.deviation > original.deviation
    assert updated.volatility == original.volatility


def test_period_updates_are_simultaneous() -> None:
    ratings = {"a": GlickoRating(1800, 50, 0.06), "b": GlickoRating(1500, 50, 0.06), "c": GlickoRating(1200, 50, 0.06)}
    updated = simultaneous_update(ratings, [("a", "b", 0.0), ("b", "c", 0.0)])
    expected_b = update_rating(ratings["b"], [(ratings["a"], 1.0), (ratings["c"], 0.0)])
    assert updated["b"] == expected_b

