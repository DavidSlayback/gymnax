from typing import Optional

from matplotlib.pyplot import Annotation, Axes

from ..environments.environment import Environment


def init_maze(ax: Axes, env: Environment, state, params: Optional) -> Annotation:
    """Initialize maze grid with agent and goal"""
    im = ax.imshow(env.occupied_map, cmap="Greys")
    anno_pos = ax.annotate(
        "A",
        fontsize=20,
        xy=(state.pos[1], state.pos[0]),
        xycoords="data",
        xytext=(state.pos[1] - 0.3, state.pos[0] + 0.25),
    )
    anno_goal = ax.annotate(
        "G",
        fontsize=20,
        xy=(state.goal[1], state.goal[0]),
        xycoords="data",
        xytext=(state.goal[1] - 0.3, state.goal[0] + 0.25),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return anno_pos


def update_maze(anim_state: Annotation, env: Environment, state) -> Annotation:
    xy = (state.pos[1], state.pos[0])

    xytext = (state.pos[1] - 0.3, state.pos[0] + 0.25)

    anim_state.set_position((xytext[0], xytext[1]))
    anim_state.xy = (xy[0], xy[1])
    return anim_state
