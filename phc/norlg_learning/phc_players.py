import os
import time

import gym
import numpy as np
import torch

from phc.norlg_learning.utils import RunningMeanStd, shape_whc_to_cwh


class CommonAgent:
    def __init__(self, config, env):
        # from rl_games.common.player import BasePlayer
        # BasePlayer.__init__(self, config)
        self.config = config
        self.env = env
        self.device = env.device

        self.observation_space = env.observation_space
        self.obs_shape = self.observation_space.shape
        self.amp_observation_space = env.amp_observation_space
        self.amp_obs_shape = self.amp_observation_space.shape

        self.num_envs = env.num_environments
        self.num_agents = env.num_agents
        self.action_space = env.action_space
        self.actions_num = self.action_space.shape[0]
        self.value_size = 1

        # TODO: move action cliping to somewhere else?
        self.clip_actions = True  # config.get("clip_actions", True)

        self.states = None  # No RNN
        self.batch_size = 1

        # TODO: check the default values? Do we need these?
        self.player_config = self.config.get("player", {})
        # self.has_central_value = self.config.get('central_value_config') is not None
        # self.render_env = self.player_config.get("render", False)
        self.games_num = self.player_config.get("games_num", 15)
        self.is_determenistic = self.player_config.get("determenistic", True)
        self.print_stats = True
        self.max_steps = 108000 // 4

        # from rl_games.algos_torch.players import PpoPlayerContinuous
        self.network = config["network"]

        # NOTE: PHC normalizes all inputs, values, amp_inputs.
        self.normalize_input = True  # self.config.get("normalize_input", False)
        self.normalize_value = True  # self.config.get("normalize_value", False)
        self._normalize_amp_input = True  # config.get("normalize_amp_input", True)
        self._build_model()

    def _build_model(self):
        obs_shape = shape_whc_to_cwh(self.obs_shape)
        config = {
            "actions_num": self.actions_num,
            "input_shape": obs_shape,
            "amp_input_shape": self.amp_obs_shape,
        }

        self.model = self.network.build(config)
        self.model.to(self.device)
        self.is_rnn = self.model.is_rnn()
        assert not self.is_rnn, "PHC policy does not use RNN"

        self.running_mean_std = RunningMeanStd(obs_shape).to(self.device) if self.normalize_input else None
        self.value_mean_std = RunningMeanStd((1,)).to(self.device) if self.normalize_value else None
        self._amp_input_mean_std = (
            RunningMeanStd(self.amp_obs_shape).to(self.device) if self._normalize_amp_input else None
        )
        self.set_eval()

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()
        if self._normalize_amp_input:
            self._amp_input_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()
        if self._normalize_amp_input:
            self._amp_input_mean_std.train()

    def get_model_weights(self):
        state_dict = {}
        state_dict["model"] = self.model.state_dict()
        if self.normalize_input:
            state_dict["running_mean_std"] = self.running_mean_std.state_dict()
        if self.normalize_value:
            state_dict["value_mean_std"] = self.value_mean_std.state_dict()
        if self._normalize_amp_input:
            state_dict["amp_input_mean_std"] = self._amp_input_mean_std.state_dict()
        return state_dict

    def set_model_weights(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(state_dict["running_mean_std"])
        if self.normalize_value and "value_mean_std" in state_dict:
            self.value_mean_std.load_state_dict(state_dict["value_mean_std"])
        if self._normalize_amp_input:
            self._amp_input_mean_std.load_state_dict(state_dict["amp_input_mean_std"])

    def restore(self, file_path):
        assert os.path.exists(file_path), "Checkpoint file does not exist"
        print("=> loading checkpoint '{}'".format(file_path))
        state_dict = torch.load(file_path)
        self.set_model_weights(state_dict)

    def env_reset(self, env_ids=None):
        obs_torch = self.env.reset(env_ids)
        # obs is already in torch
        return obs_torch

    def _preproc_obs(self, obs_batch):
        if self.normalize_input:
            obs_batch = self.running_mean_std(obs_batch)
        return obs_batch

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs


class PHCPlayer(CommonAgent):
    def __init__(self, config, env_creator):
        env_config = config.get("env_config", {})
        env = env_creator(**env_config)
        super().__init__(config, env)

        # NOTE: Relax the early termination condition
        self.env.task._termination_distances[:] = 0.5

    def run(self):
        n_games = self.games_num  # necessary?
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        games_played = 0

        cr = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        steps = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        done_indices = None

        for _ in range(n_games):
            if games_played >= n_games:
                break

        with torch.no_grad():
            for _ in range(self.max_steps):
                obs_torch = self.env_reset(done_indices)

                action = self.get_action(obs_torch, is_determenistic)

                """Stepping the environment"""
                obs_torch, r, done, info = self.env.step(action)

                cr += r
                steps += 1

                # NOTE: _post_step does batch eval
                # done = self._post_step(info, done.clone())

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[:: self.num_agents].flatten()
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()
                    print(
                        "reward:",
                        cur_rewards / done_count,
                        "steps:",
                        cur_steps / done_count,
                    )

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

    def get_action(self, obs_torch, is_determenistic=False):
        obs_torch = self._preproc_obs(obs_torch)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs_torch,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict["mus"]
        action = res_dict["actions"]

        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        # current_action = current_action.detach()

        if self.clip_actions:
            return torch.clamp(current_action, -1.0, 1.0)
        else:
            return current_action
