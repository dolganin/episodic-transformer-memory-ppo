import gymnasium as gym
import numpy as np
import time

from gymnasium import spaces
from minigrid.wrappers import RGBImgPartialObsWrapper, ViewSizeWrapper, ImgObsWrapper


class Minigrid:
    def __init__(self, name: str):
        self._env = gym.make(name)

        # Decrease the agent's view size to raise the agent's memory challenge
        if "Memory" in name:
            view_size = 3
            self.tile_size = 28
            hw = view_size * self.tile_size
            self.max_episode_steps = 96
            self._action_space = spaces.Discrete(3)
            self._env = ViewSizeWrapper(self._env, view_size)
            self._env = RGBImgPartialObsWrapper(self._env, tile_size=self.tile_size)
        else:
            view_size = 7
            self.tile_size = 8
            hw = view_size * self.tile_size
            self.max_episode_steps = 64
            self._action_space = self._env.action_space
            self._env = ViewSizeWrapper(self._env, view_size)
            self._env = RGBImgPartialObsWrapper(self._env, tile_size=self.tile_size)

        self._env = ImgObsWrapper(self._env)

        self._observation_space = spaces.Box(
            low=0,
            high=1.0,
            shape=(3, hw, hw),
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
        obs = obs.astype(np.float32) / 255.0
        # channel-first
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        return obs

    def step(self, action):
        result = self._env.step(action[0])
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        self._rewards.append(reward)
        obs = obs.astype(np.float32) / 255.0

        if self.t == self.max_episode_steps - 1:
            done = True

        if done:
            info = {"reward": sum(self._rewards), "length": len(self._rewards)}
        else:
            info = None

        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        self.t += 1
        return obs, reward, done, info

    def render(self):
        img = self._env.render(tile_size=96)
        time.sleep(0.5)
        return img

    def close(self):
        self._env.close()
