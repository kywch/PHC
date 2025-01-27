from gym import spaces
import numpy as np
import torch


# NOTE: some vars are kept for compatibility with the rlgames-based training code
# TODO: clean up for pufferl
class VecTaskWrapper:
    def __init__(self, task, rl_device, clip_observations=5.0):
        self.task = task
        self.rl_device = rl_device
        self.clip_obs = clip_observations

        self.num_environments = task.num_envs
        self.num_agents = 1  # used for multi-agent environments
        self.num_observations = task.num_obs
        self.num_states = task.num_states
        self.num_actions = task.num_actions

        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.state_space = spaces.Box(
            np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf
        )
        self.act_space = spaces.Box(
            np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0
        )
        self._amp_obs_space = spaces.Box(
            np.ones(task.get_num_amp_obs()) * -np.Inf, np.ones(task.get_num_amp_obs()) * np.Inf
        )

    def get_number_of_agents(self):
        return self.num_agents

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    def _clip_buffer(self, buffer):
        return torch.clamp(buffer, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def reset(self, env_ids=None):
        self.task.reset(env_ids)
        return self._clip_buffer(self.task.obs_buf)

    def get_state(self):
        return self._clip_buffer(self.task.states_buf)

    def step(self, actions):
        self.task.step(actions)

        obs = self._clip_buffer(self.task.obs_buf)
        rew = self.task.rew_buf.to(self.rl_device)
        done = self.task.reset_buf.to(self.rl_device)
        info = self.task.extras
        return obs, rew, done, info

    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)
