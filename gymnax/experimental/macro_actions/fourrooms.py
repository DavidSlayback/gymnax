# Macro-actions for four rooms domain
from typing import Sequence, Any, Tuple

import chex
import jax.lax
import jax.numpy as jnp
import numpy as np
from ...environments.misc.fourrooms import FourRooms, numbered_four_rooms_map


def create_original_hallway_options() -> Sequence[Any]:
    """Create option for each hallway in fourrooms, leading to it from anywhere in adjoining two rooms"""
    directions = np.array(FourRooms.directions)  # directions don't vary
    numbered_map = numbered_four_rooms_map  # Number for each room, 0 for each hallway
    hallways_coords = np.argwhere(numbered_map == 0)  # 4x2 row,col coords of each hallway
    hallway_adj = numbered_map[tuple((hallways_coords[:, None] + directions[None, :]).transpose(-1, 0, 1))]  # [H,D]
    hallway_adjoining_rooms = hallway_adj[hallway_adj > 0].reshape(4, -1)  # [H, 2] adjoining rooms for each hallway
    # For each hallway, need to add hallway to set of
    options = []
    for i, (c, rs) in enumerate(zip(hallways_coords, hallway_adjoining_rooms)):
        valid_init_states = np.argwhere(np.isin(numbered_map, rs))
        valid_init_states = np.concatenate([valid_init_states, hallways_coords[np.isin(hallway_adjoining_rooms, rs).any(-1) & ~np.isin(hallway_adjoining_rooms, rs).all(-1)]], 0)
        pi = np.zeros(numbered_map.shape + (directions.shape[0],))  # Default invalid action
        beta = np.zeros(numbered_map.shape)  # Default non-terminal
        beta[c[0], c[1]] = 1.  # Terminate
        # One-step lookahead is fine. Check all possible moves for closest manhattan distance. Sample one at random
        for c_o in valid_init_states:
            c_adj = c_o + directions
            d = np.linalg.norm(c_adj - c, axis=-1, ord=1)
            d_mask = d == d.min()
            pi[c_o[0], c_o[1], d_mask] = 1. / d_mask.sum()
        options.append((valid_init_states, pi, beta))
    return options


def onestep_manhattan_macro(c: chex.Array, goal: chex.Array) -> Tuple[int, bool]:
    """Given a goal that can be reached without worrying about obstacles go towards it

    Useful for moving hallway to nearest hallway (e.g., E -> S)
    Will fail if route requires more than one-step lookahead
    (e.g., E -> W has to make moves that temporarily increase manhattan distance)
    Deterministic action (argmin) and termination (only at goal)

    Args:
        c: Current (row, col) coordinate
        goal: Goal (row, col) coordinate
    Returns:
        a: Next action -1 no action, 0-N, 1-E, 2-W, 3-S
        t: Termination if actually at goal (just planning to take action could fail)
    """
    t = jax.lax.select((c == goal).all(-1), True, False)
    a = jax.lax.select(t, -1, jnp.linalg.norm((c + FourRooms.directions) - goal, axis=-1, ord=1).argmin(-1))
    return a, t





