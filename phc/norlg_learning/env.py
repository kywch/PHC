from gym import spaces
import numpy as np
import torch


# This wrapper combines VecTask, VecTaskPython, VecTaskPythonWrapper, RLGPUEnvWrapper
# Also does the action clipping
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

        if hasattr(self.task, "single_observation_space"):
            self.obs_space = self.task.single_observation_space
        else:
            self.obs_space = spaces.Box(
                np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf, dtype=np.float32
            )

        if hasattr(self.task, "amp_observation_space"):
            self._amp_obs_space = self.task.amp_observation_space
        else:
            num_amp_obs = self.task.get_num_amp_obs()
            self._amp_obs_space = spaces.Box(
                np.ones(num_amp_obs) * -np.Inf, np.ones(num_amp_obs) * np.Inf, dtype=np.float32
            )

        if hasattr(self.task, "single_action_space"):
            self.act_space = self.task.single_action_space
        else:
            self.act_space = spaces.Box(
                np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0, dtype=np.float32
            )

        self.state_space = spaces.Box(
            np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf, dtype=np.float32
        )

        # RLGPU env wrapper
        self.add_state_obs = self.task.num_states > 0
        self.full_state = {}
        self.reset()

    def get_number_of_agents(self):
        return self.num_agents

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    @property
    def action_space(self):
        return self.act_space

    # @property
    # def num_envs(self):
    #     return self.num_environments

    # @property
    # def num_acts(self):
    #     return self.num_actions

    # @property
    # def num_obs(self):
    #     return self.num_observations

    def _clip_obs(self, buffer):
        return torch.clamp(buffer, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def reset(self, env_ids=None):
        obs = self.task.reset(env_ids)
        if obs is None:
            obs = self.task.obs_buf
        self.full_state["obs"] = self._clip_obs(obs)

        if self.add_state_obs:
            self.full_state["states"] = self.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def step(self, actions):
        obs, rewards, dones, infos = self.task.step(actions)

        if obs is None:
            obs = self.task.obs_buf
            rewards = self.task.rew_buf.to(self.rl_device)
            dones = self.task.reset_buf.to(self.rl_device)
            infos = self.task.extras
        obs = self._clip_obs(obs)

        if self.add_state_obs:
            self.full_state["obs"] = obs
            self.full_state["states"] = self.get_state()
            return self.full_state, rewards, dones, infos
        else:
            return obs, rewards, dones, infos

    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)

    def get_state(self):
        return self._clip_obs(self.task.states_buf)
