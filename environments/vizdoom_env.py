import numpy as np
from gymnasium import spaces


class VizDoom:
    """Minimal VizDoom wrapper yielding CHW float32 observations."""

    def __init__(self, config: dict):
        import vizdoom as vzd

        self.game = vzd.DoomGame()
        if "config" in config:
            self.game.load_config(config["config"])
        if "scenario" in config:
            self.game.set_doom_scenario_path(config["scenario"])
        resolution = config.get("resolution")
        if resolution is not None:
            self.game.set_screen_resolution(getattr(vzd.ScreenResolution, resolution))
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.init()

        self.frame_skip = int(config.get("frame_skip", 1))
        n_buttons = self.game.get_available_buttons_size()
        self.actions = np.eye(n_buttons, dtype=np.uint8)

        obs = self._get_obs()
        self._observation_space = spaces.Box(low=0, high=1.0, shape=obs.shape, dtype=np.float32)
        self._action_space = spaces.Discrete(len(self.actions))
        self.max_episode_steps = self.game.get_episode_timeout() or 0
        self.t = 0
        self._rewards = []

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _get_obs(self):
        state = self.game.get_state()
        if state is None:
            shape = (self.game.get_screen_channels(), self.game.get_screen_height(), self.game.get_screen_width())
            return np.zeros(shape, dtype=np.float32)
        buf = state.screen_buffer
        if buf.ndim == 3 and buf.shape[0] in (1, 3, 4):
            arr = buf
        else:
            arr = np.transpose(buf, (2, 0, 1))
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        return arr

    def reset(self):
        self.game.new_episode()
        self.t = 0
        self._rewards = []
        return self._get_obs()

    def step(self, action):
        reward = self.game.make_action(self.actions[int(action)].tolist(), self.frame_skip)
        self._rewards.append(reward)
        done = self.game.is_episode_finished() or (self.max_episode_steps and self.t >= self.max_episode_steps - 1)
        obs = self._get_obs() if not done else self._observation_space.low.copy()
        info = {"reward": sum(self._rewards), "length": len(self._rewards)} if done else None
        self.t += 1
        return obs, reward, done, info

    def render(self):
        state = self.game.get_state()
        if state is None:
            return np.zeros((self._observation_space.shape[1], self._observation_space.shape[2], 3), dtype=np.uint8)
        buf = state.screen_buffer
        if buf.ndim == 3 and buf.shape[0] in (1, 3, 4):
            arr = np.transpose(buf, (1, 2, 0))
        else:
            arr = buf
        return arr

    def close(self):
        self.game.close()
