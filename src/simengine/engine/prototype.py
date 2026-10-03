"""Prototype three-species ecosystem (grass, hare, fox) in the planned layout.

Throwaway scope: one cell, one age class. Species axis, cell axis and age axis
already follow docs/DECISIONS.md item 6 so Phase 2 can generalize this.

food_web[prey, predator] is the conversion efficiency: the fraction of eaten
prey biomass that becomes predator biomass. Zero means no feeding link.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

TICK_YEARS = 0.25  # decision 1: one tick is one season

Array = NDArray[np.float64]

GRASS, HARE, FOX = 0, 1, 2
SPECIES_NAMES = ("grass", "hare", "fox")


@dataclass(frozen=True, eq=False)
class State:
    """Simulation state. Arrays are never mutated in place."""

    biomass: Array  # [S, C, A] kg per cell per age class
    trait_mean: Array  # [S, C, T] unused until Phase 5
    trait_var: Array  # [S, C, T] unused until Phase 5
    food_web: Array  # [S, S] efficiency, indexed [prey, predator]
    active: NDArray[np.bool_]  # [S]


@dataclass(frozen=True, eq=False)
class Params:
    """Per-species rates are per year. Matrices are indexed [prey, predator]."""

    growth_rate: Array  # [S] r, intrinsic growth of producers (0 for consumers)
    carrying_capacity: Array  # [S] K in kg per cell, must be > 0
    attack: Array  # [S, S] a, kg of prey found per kg predator per year per kg prey
    handling: Array  # [S] h, years spent handling one kg of prey per kg predator
    respiration: Array  # [S] m, fraction of own biomass burned per year


def validate(state: State, params: Params) -> None:
    s, c, a = state.biomass.shape
    if a != 1:
        raise ValueError("prototype supports exactly one age class")
    if state.food_web.shape != (s, s):
        raise ValueError("food_web must be [S, S]")
    if state.active.shape != (s,):
        raise ValueError("active must be [S]")
    if state.trait_mean.shape[:2] != (s, c) or state.trait_var.shape[:2] != (s, c):
        raise ValueError("trait arrays must be [S, C, T]")
    if np.any(state.biomass < 0) or not np.all(np.isfinite(state.biomass)):
        raise ValueError("biomass must be finite and non-negative")
    if np.any(state.food_web < 0) or np.any(state.food_web > 1):
        raise ValueError("food_web efficiencies must be in [0, 1]")
    if params.attack.shape != (s, s):
        raise ValueError("attack must be [S, S]")
    for name in ("growth_rate", "carrying_capacity", "handling", "respiration"):
        if getattr(params, name).shape != (s,):
            raise ValueError(f"{name} must be [S]")
    if np.any(params.carrying_capacity <= 0):
        raise ValueError("carrying_capacity must be positive")
    if np.any(params.growth_rate < 0) or np.any(params.attack < 0):
        raise ValueError("rates must be non-negative")
    if np.any(params.handling < 0) or np.any(params.respiration < 0):
        raise ValueError("rates must be non-negative")


def _grow(biomass: Array, params: Params) -> Array:
    """Exact logistic solution over one tick.

    B(t+dt) = K * B * e / (K + B * (e - 1)),  e = exp(r * dt)
    Positive for any B >= 0, never overshoots K, equals B when r = 0.
    """
    k = params.carrying_capacity[:, None]
    e = np.exp(params.growth_rate * TICK_YEARS)[:, None]
    return k * biomass * e / (k + biomass * (e - 1.0))


def _feed(biomass: Array, state: State, params: Params) -> Array:
    """Biomass change from feeding for one tick, shape [S, C].

    Holling type II, multi-prey, biomass based:
      intake of prey i per kg of predator j = a_ij B_i / (1 + h_j * sum_k a_kj B_k)
    Total eaten of prey i by j = intake * B_j * dt, then capped so that all
    predators together never eat more than the prey biomass that exists.
    """
    b = biomass
    encounter = np.einsum("ij,ic->jc", params.attack, b)  # [S_pred, C]
    denom = 1.0 + params.handling[:, None] * encounter  # [S_pred, C]
    eaten = (
        params.attack[:, :, None]
        * b[:, None, :]
        * b[None, :, :]
        / denom[None, :, :]
        * TICK_YEARS
    )  # [S_prey, S_pred, C]
    total = eaten.sum(axis=1)  # [S_prey, C]
    scale = np.ones_like(total)
    over = total > b
    scale[over] = b[over] / total[over]
    eaten = eaten * scale[:, None, :]
    lost = eaten.sum(axis=1)  # [S_prey, C]
    gained = np.einsum("ij,ijc->jc", state.food_web, eaten)  # [S_pred, C]
    return gained - lost


def _respire(biomass: Array, params: Params) -> Array:
    """Exact exponential decay of own biomass over one tick."""
    return biomass * np.exp(-params.respiration * TICK_YEARS)[:, None]


def tick(state: State, params: Params) -> State:
    """Advance one season. Pure: returns a new State, never mutates the input.

    Fixed order (operator splitting): growth, then feeding, then respiration.
    Each stage keeps biomass non-negative, so the result always is.
    """
    validate(state, params)
    b = state.biomass[:, :, 0] * state.active[:, None]
    b = _grow(b, params)
    b = b + _feed(b, state, params)
    b = np.maximum(b, 0.0)  # removes rounding noise like -1e-17 only
    b = _respire(b, params)
    return replace(state, biomass=b[:, :, None])


def default_world() -> tuple[State, Params]:
    """Provisional constants. Step 1.5 tunes them for coexistence."""
    s = 3
    food_web = np.zeros((s, s))
    food_web[GRASS, HARE] = 0.30
    food_web[HARE, FOX] = 0.10
    attack = np.zeros((s, s))
    attack[GRASS, HARE] = 0.01
    attack[HARE, FOX] = 0.05
    params = Params(
        growth_rate=np.array([1.0, 0.0, 0.0]),
        carrying_capacity=np.array([1000.0, 1.0, 1.0]),
        attack=attack,
        handling=np.array([0.0, 0.1, 0.1]),
        respiration=np.array([0.0, 0.5, 0.2]),
    )
    state = State(
        biomass=np.array([500.0, 20.0, 2.0]).reshape(s, 1, 1),
        trait_mean=np.zeros((s, 1, 1)),
        trait_var=np.zeros((s, 1, 1)),
        food_web=food_web,
        active=np.ones(s, dtype=bool),
    )
    return state, params