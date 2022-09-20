from typing import Optional

import numpy as np
from matplotlib.image import AxesImage
from matplotlib.pyplot import Axes

from ..environments.environment import Environment


def init_minatar(ax: Axes, env: Environment, state, params: Optional) -> AxesImage:
    """Initialize MinAtar rendering window"""
    import seaborn as sns
    import matplotlib.colors as colors

    obs = env.get_obs(state)
    n_channels = env.obs_shape[-1]
    # The seaborn color_palette cubhelix is used to assign visually distinct colors to each channel for the env
    cmap = sns.color_palette("cubehelix", n_channels)
    cmap.insert(0, (0, 0, 0))
    cmap = colors.ListedColormap(cmap)
    bounds = [i for i in range(n_channels + 2)]
    norm = colors.BoundaryNorm(bounds, n_channels + 1)
    numerical_state = (
        np.amax(obs * np.reshape(np.arange(n_channels) + 1, (1, 1, -1)), 2)
        + 0.5
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return ax.imshow(
        numerical_state, cmap=cmap, norm=norm, interpolation="none"
    )


def update_minatar(anim_state: AxesImage, env: Environment, state) -> AxesImage:
    """Update positions of entities"""
    obs = env.get_obs(state)
    n_channels = env.obs_shape[-1]
    numerical_state = (
        np.amax(obs * np.reshape(np.arange(n_channels) + 1, (1, 1, -1)), 2)
        + 0.5
    )
    anim_state.set_data(numerical_state)
    return anim_state
