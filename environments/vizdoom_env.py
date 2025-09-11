import gymnasium as gym
import numpy as np

from gymnasium import spaces


class VizDoom:
    """Wrapper for VizDoom environments producing CHW float32 observations."""

    def __init__(self, name: str):
        self._env = gym.make(name, render_mode="rgb_array")

        self._action_space = self._env.action_space
        self.max_episode_steps = getattr(self._env.spec, "max_episode_steps", 0)

        obs_shape = self._env.observation_space.shape  # (H, W, C)
        self._observation_space = spaces.Box(
            low=0,
            high=1.0,
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
            dtype=np.float32,
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        seed = int(np.random.randint(0, 2**31 - 1))
        obs, info = self._env.reset(seed=seed)
        self.t = 0
        self._rewards = []

        obs = obs.astype(np.float32)
        if obs.max() > 1.0:
            obs /= 255.0
        obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW
        return obs

    def step(self, action):
        result = self._env.step(action[0]) if isinstance(action, (list, np.ndarray)) else self._env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        self._rewards.append(reward)
        obs = obs.astype(np.float32)
        if obs.max() > 1.0:
            obs /= 255.0
        obs = np.transpose(obs, (2, 0, 1))

        if self.t == self.max_episode_steps - 1:
            done = True

        if done:
            info = {"reward": sum(self._rewards), "length": len(self._rewards)}
        else:
            info = None

        self.t += 1
        return obs, reward, done, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

