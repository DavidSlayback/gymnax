import gym
import numpy as np
from matplotlib.image import AxesImage
from matplotlib.pyplot import Axes

from ..environments.environment import Environment


def set_gym_params(gym_env: gym.Env, env_name: str, params) -> None:
    """Set gym.Env instance parameters from EnvParams"""
    if env_name == "Acrobot-v1":
        gym_env.env.LINK_LENGTH_1 = params.link_length_1
        gym_env.env.LINK_LENGTH_2 = params.link_length_2
    elif env_name == "CartPole-v1":
        gym_env.env.x_threshold = params.x_threshold
        gym_env.env.length = params.length
    elif env_name == "Pendulum-v1":
        pass
    elif env_name == "MountainCar-v0":
        gym_env.env.max_position = params.max_position
        gym_env.env.min_position = params.min_position
        gym_env.env.goal_position = params.goal_position
    elif env_name == "MountainCarContinuous-v0":
        gym_env.env.max_position = params.max_position
        gym_env.env.min_position = params.min_position
        gym_env.env.goal_position = params.goal_position
    return


def get_gym_state(state, env_name) -> np.ndarray:
    """Get gym.Env state equivalent from Gymnax EnvState"""
    if env_name == "Acrobot-v1":
        return np.array(
            [
                state.joint_angle1,
                state.joint_angle2,
                state.velocity_1,
                state.velocity_2,
            ]
        )
    elif env_name == "CartPole-v1":
        return np.array([state.x, state.x_dot, state.theta, state.theta_dot])
    elif env_name == "Pendulum-v1":
        return np.array([state.theta, state.theta_dot, state.last_u])
    elif env_name == "MountainCar-v0":
        return np.array([state.position, state.velocity])
    elif env_name == "MountainCarContinuous-v0":
        return np.array([state.position, state.velocity])


def init_gym(ax: Axes, env: Environment, state, params) -> AxesImage:
    """Use gym.Env rendering to render gymnax environment"""
    if env.name == "Pendulum-v1":
        gym_env = gym.make("Pendulum-v0")
    else:
        gym_env = gym.make(env.name)
    gym_env.reset()
    set_gym_params(gym_env, env.name, params)
    gym_state = get_gym_state(state, env.name)
    if env.name == "Pendulum-v1":
        gym_env.env.last_u = gym_state[-1]
    gym_env.env.state = gym_state
    rgb_array = gym_env.render(mode="rgb_array")
    ax.set_xticks([])
    ax.set_yticks([])
    gym_env.close()
    return ax.imshow(rgb_array)


def update_gym(anim_state: AxesImage, env: Environment, state) -> AxesImage:
    if env.name == "Pendulum-v1":
        gym_env = gym.make("Pendulum-v0")
    else:
        gym_env = gym.make(env.name)
    gym_state = get_gym_state(state, env.name)
    if env.name == "Pendulum-v1":
        gym_env.env.last_u = gym_state[-1]
    gym_env.env.state = gym_state
    rgb_array = gym_env.render(mode="rgb_array")
    anim_state.set_data(rgb_array)
    gym_env.close()
    return anim_state
