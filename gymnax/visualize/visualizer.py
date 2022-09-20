from typing import Optional
from typing import Sequence, Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from .vis_catch import init_catch, update_catch
from .vis_circle import init_circle, update_circle
from .vis_gym import init_gym, update_gym
from .vis_maze import init_maze, update_maze
from .vis_minatar import init_minatar, update_minatar
from ..environments.environment import Environment, EnvState, EnvParams
from ..registration import registered_envs

# Specialized animation init and update functions for each environment
gym_envs = ["Acrobot-v1", "CartPole-v1", "Pendulum-v1", "MountainCar-v0", "MountainCarContinuous-v0"]
maze_envs = ["MetaMaze-misc", "FourRooms-misc"]
minatar_envs = [env for env in registered_envs if 'MinAtar' in env]
init_and_update_fns = {
    **dict.fromkeys(gym_envs, (init_gym, update_gym)),
    **dict.fromkeys(minatar_envs, (init_minatar, update_minatar)),
    **dict.fromkeys(maze_envs, (init_maze, update_maze)),
    "PointRobot-misc": (init_circle, update_circle),
    "Catch-bsuite": (init_catch, update_catch),
}


class Visualizer:
    def __init__(self, env: Environment, env_params: EnvParams, state_seq: Sequence[EnvState], reward_seq: Optional[Sequence[float]] = None):
        self.env = env
        self.env_params = env_params
        self.state_seq = state_seq
        self.reward_seq = reward_seq
        self.anim_state: Any = None  # Most likely AxesImage, but can be Tuple (see vis_circle)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 5))
        if env.name not in gym_envs:
            self.interval = 100
        else:
            import gym
            assert gym.__version__ == "0.19.0"  # use older gym and pyglet
            self.interval = 50
        self._init_fn, self._update_fn = init_and_update_fns[env.name]

    def animate(
        self,
        save_fname: Optional[str] = "test.gif",
        view: bool = False,
    ):
        """Anim for 2D fct - x (#steps, #pop, 2) & fitness (#steps, #pop)"""
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.state_seq),
            init_func=self.init,
            blit=False,
            interval=self.interval,
        )
        # Save the animation to a gif
        if save_fname is not None:
            ani.save(save_fname)
        # Simply view it 3 times
        if view:
            plt.show(block=False)
            plt.pause(3)
            plt.close()

    def init(self):
        # Plot placeholder points
        self.anim_state = self._init_fn(self.ax, self.env, self.state_seq[0], self.env_params)
        self.fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.95])

    def update(self, frame: int):
        self.anim_state = self._update_fn(self.anim_state, self.env, self.state_seq[frame])
        if self.reward_seq is None:
            self.ax.set_title(
                f"{self.env.name} - Step {frame + 1}", fontsize=15
            )
        else:
            self.ax.set_title(
                "{}: Step {:4.0f} - Return {:7.2f}".format(
                    self.env.name, frame + 1, self.reward_seq[frame]
                ),
                fontsize=15,
            )
