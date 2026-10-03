from dataclasses import replace

import numpy as np
import pytest

from simengine.engine import prototype as proto
from simengine.engine.prototype import FOX, GRASS, HARE, default_world, tick


def run(state, params, ticks):
    for _ in range(ticks):
        state = tick(state, params)
    return state


def test_tick_is_pure_and_deterministic() -> None:
    state, params = default_world()
    before = state.biomass.copy()
    first = tick(state, params)
    second = tick(state, params)
    assert np.array_equal(state.biomass, before)
    assert np.array_equal(first.biomass, second.biomass)


def test_biomass_stays_non_negative_and_finite() -> None:
    state, params = default_world()
    for _ in range(800):
        state = tick(state, params)
        assert np.all(state.biomass >= 0)
        assert np.all(np.isfinite(state.biomass))


def test_producer_converges_to_carrying_capacity() -> None:
    state, params = default_world()
    state = replace(state, biomass=np.array([10.0, 0.0, 0.0]).reshape(3, 1, 1))
    final = run(state, params, 400)
    k = params.carrying_capacity[GRASS]
    assert final.biomass[GRASS, 0, 0] == pytest.approx(k, rel=1e-3)
    assert np.all(final.biomass[:, 0, 0][[HARE, FOX]] == 0)


def test_logistic_never_overshoots_capacity_from_above() -> None:
    state, params = default_world()
    k = params.carrying_capacity[GRASS]
    state = replace(state, biomass=np.array([5 * k, 0.0, 0.0]).reshape(3, 1, 1))
    for _ in range(40):
        state = tick(state, params)
        assert state.biomass[GRASS, 0, 0] >= k * (1 - 1e-12)


def test_consumers_without_food_decay() -> None:
    state, params = default_world()
    state = replace(state, biomass=np.array([0.0, 10.0, 5.0]).reshape(3, 1, 1))
    final = run(state, params, 200)
    assert np.all(final.biomass[:, 0, 0][[HARE, FOX]] < 0.01)


def test_feeding_cannot_eat_more_than_exists() -> None:
    state, params = default_world()
    # Huge predator, tiny prey, no regrowth: prey must not go negative.
    params = replace(params, growth_rate=np.zeros(3))
    state = replace(state, biomass=np.array([1.0, 500.0, 0.0]).reshape(3, 1, 1))
    after = tick(state, params)
    assert after.biomass[GRASS, 0, 0] >= 0.0
    assert after.biomass[GRASS, 0, 0] == pytest.approx(0.0, abs=1e-9)


def test_feeding_conserves_mass_with_perfect_efficiency() -> None:
    state, params = default_world()
    food_web = np.zeros((3, 3))
    food_web[GRASS, HARE] = 1.0
    params = replace(
        params,
        growth_rate=np.zeros(3),
        respiration=np.zeros(3),
    )
    state = replace(
        state,
        food_web=food_web,
        biomass=np.array([100.0, 10.0, 0.0]).reshape(3, 1, 1),
    )
    after = tick(state, params)
    assert after.biomass.sum() == pytest.approx(state.biomass.sum(), rel=1e-12)
    assert after.biomass[HARE, 0, 0] > 10.0


def test_inactive_species_stay_at_zero() -> None:
    state, params = default_world()
    active = np.array([True, False, True])
    state = replace(state, active=active)
    after = tick(state, params)
    assert after.biomass[HARE, 0, 0] == 0.0


def test_default_world_coexists_for_200_years() -> None:
    state, params = default_world()
    low = np.full(3, np.inf)
    for _ in range(800):
        state = tick(state, params)
        low = np.minimum(low, state.biomass[:, 0, 0])
    assert np.all(low > 1.0)


def test_validation_rejects_bad_input() -> None:
    state, params = default_world()
    bad_eff = state.food_web.copy()
    bad_eff[GRASS, HARE] = 1.5
    with pytest.raises(ValueError):
        tick(replace(state, food_web=bad_eff), params)
    with pytest.raises(ValueError):
        tick(replace(state, biomass=-state.biomass), params)
    with pytest.raises(ValueError):
        tick(replace(state, biomass=np.zeros((3, 1, 2))), params)
    with pytest.raises(ValueError):
        tick(state, replace(params, carrying_capacity=np.zeros(3)))


def test_tick_length_is_one_season() -> None:
    assert proto.TICK_YEARS == 0.25