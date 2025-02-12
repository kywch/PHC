from isaacgym import gymapi

from gym import spaces
import numpy as np
import torch

from phc.pufferl.humanoid_phc import HumanoidPHC
from phc.pufferl.render_env import HumanoidRenderEnv


def create_rlgpu_env(cfg, **kwargs):
    task_cls = HumanoidPHC
    if cfg.test and not cfg.headless:
        task_cls = HumanoidRenderEnv

    task = task_cls(
        cfg=cfg,
        sim_params=None,
        physics_engine=gymapi.SIM_PHYSX,
        device_type=cfg.device,
        device_id=cfg.device_id,
        headless=cfg.headless,
    )

    env = VecTaskWrapper(task)

    print(env.num_environments)
    print(env.num_actions)
    print(env.num_observations)
    print(env.num_states)

    return env


# This wrapper combines VecTask, VecTaskPython, VecTaskPythonWrapper, RLGPUEnvWrapper
class VecTaskWrapper:
    def __init__(self, task, clip_observations=None, clip_actions=True):
        self.task = task
        self.clip_obs = clip_observations
        self.clip_actions = clip_actions

        self.num_environments = task.num_envs
        self.num_agents = 1  # used for multi-agent environments
        self.num_observations = task.num_obs
        self.num_states = task.num_states
        self.num_actions = task.num_actions

        if hasattr(self.task, "single_observation_space"):
            self.observation_space = self.task.single_observation_space
        else:
            self.observation_space = spaces.Box(
                np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf, dtype=np.float32
            )

        if hasattr(self.task, "amp_observation_space"):
            self.amp_observation_space = self.task.amp_observation_space
        else:
            num_amp_obs = self.task.get_num_amp_obs()
            self.amp_observation_space = spaces.Box(
                np.ones(num_amp_obs) * -np.Inf, np.ones(num_amp_obs) * np.Inf, dtype=np.float32
            )

        if hasattr(self.task, "single_action_space"):
            self.action_space = self.task.single_action_space
        else:
            self.action_space = spaces.Box(
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
    def device(self):
        return self.task.device

    def _clip_obs(self, buffer):
        if self.clip_obs is None:
            return buffer

        return torch.clamp(buffer, -self.clip_obs, self.clip_obs)

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
        if self.clip_actions:
            actions = torch.clamp(actions, -1.0, 1.0)

        obs, rewards, dones, infos = self.task.step(actions)

        if obs is None:
            obs = self.task.obs_buf
            # NOTE: simple rew normalization
            rewards = self.task.rew_buf.clone() / 100.0
            dones = self.task.reset_buf
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
