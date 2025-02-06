import os
import time
import shutil
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch import optim

import wandb
from tensorboardX import SummaryWriter

from phc.norlg_learning.utils import RunningMeanStd, AMPDataset, AverageMeter, ExperienceBuffer, ReplayBuffer


def mean_list(val):
    return torch.mean(torch.stack(val))


def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma, reduce=True):
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    if reduce:
        return kl.mean()
    else:
        return kl


class PHCAgent:
    def __init__(self, config, env_creator):
        env_config = config.get("env_config", {})
        env = env_creator(**env_config)

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

        # Set is_train to True for training
        self.is_train = False

    @property
    def task_env(self):
        return self.env.task

    def _build_model(self):
        config = {
            "actions_num": self.actions_num,
            "input_shape": self.obs_shape,
            "amp_input_shape": self.amp_obs_shape,
        }

        self.model = self.network.build(config)
        self.model.to(self.device)
        self.is_rnn = self.model.is_rnn()
        assert not self.is_rnn, "PHC policy does not use RNN"

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device) if self.normalize_input else None
        self.running_mean_std_temp = deepcopy(self.running_mean_std)

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

    def _preproc_obs(self, obs_batch, use_temp=False):
        # NOTE: PHC uses the frozen running mean/std during training.
        # TODO: Compare if use_temp is needed.
        if self.normalize_input:
            if use_temp:
                # Just updating the running mean/std
                self.running_mean_std(obs_batch)

                # Return the norm obs using the frozen running mean/std
                obs_batch = self.running_mean_std_temp(obs_batch)

            else:
                obs_batch = self.running_mean_std(obs_batch)

        return obs_batch

    #####################################################################
    ### Play: IMAMPPlayerContinuous.run() in im_amp_players.py
    #####################################################################

    def play(self):
        self.set_eval()

        # NOTE: Relax the early termination condition
        self.task_env._termination_distances[:] = 0.5

        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        games_played = 0

        cr = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        steps = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        done_indices = []

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
        processed_obs = self._preproc_obs(obs_torch)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": processed_obs,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict["mus"]
        action = res_dict["actions"]

        if is_determenistic:
            current_action = mu
        else:
            current_action = action

        return current_action

    #####################################################################
    ### Training: IMAmpAgent/AMPAgent/CommonAgent.train()
    #####################################################################

    def config_train(self):
        self.use_action_masks = False  # config.get('use_action_masks', False)
        self.is_train = True  # config.get('is_train', True)
        self.ppo = True  # config['ppo']
        self.save_freq = self.config.get("save_frequency", 0)
        self.max_epochs = self.config.get("max_epochs", 0)

        self.num_actors = self.config["num_actors"]
        self.algo_observer = self.config.get("algo_observer", None)

        # Using default reward shaper, which does not do anything
        self.rewards_shaper = self.config["reward_shaper"]  # CHECK ME

        self.network_path = self.config.get("network_path", "./nn/")
        self.log_path = self.config.get("log_path", "runs/")

        # Reward weights
        self._task_reward_w = self.config.get("task_reward_w", 0.5)
        self._disc_reward_w = self.config.get("disc_reward_w", 0.5)

        # TODO: Test this, and if it doesn't make difference, remove it?
        # use temp running mean to make sure the obs used for training is the same as calc gradient.
        self.temp_running_mean = self.task_env.temp_running_mean

        # PPO-related
        self.e_clip = self.config["e_clip"]  # 0.2
        self.clip_value = self.config["clip_value"]  # CHECK ME: this is False
        self.horizon_length = self.config["horizon_length"]  # 32
        self.normalize_advantage = self.config["normalize_advantage"]  # True
        self.gamma = self.config["gamma"]  # 0.99
        self.tau = self.config["tau"]  # 0.95

        # PHC uses gradient clipping
        self.truncate_grads = True  # self.config.get('truncate_grads', False)
        self.grad_norm = self.config["grad_norm"]  # 50

        # But the grad_norm seems to be off... 
        # TODO: Remove this
        self.mixed_precision = self.config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # Loss weights
        self.critic_coef = self.config.get("critic_coef", 5.0)
        self.entropy_coef = self.config.get("entropy_coef", 0.0)
        self.bounds_loss_coef = self.config.get("bounds_loss_coef", 10.0)
        self._disc_coef = self.config.get("disc_coef", 5.0)

        # Discriminator-related
        self._disc_logit_reg = self.config["disc_logit_reg"]  # 0.01
        self._disc_grad_penalty = self.config["disc_grad_penalty"]  # 5
        self._disc_weight_decay = self.config["disc_weight_decay"]  # 0.0001
        self._disc_reward_scale = self.config["disc_reward_scale"]  # 2

        # self.games_num = self.config['minibatch_size'] // self.seq_len # it is used only for current rnn implementation
        self.batch_size = self.horizon_length * self.num_actors * self.num_agents
        self.batch_size_envs = self.horizon_length * self.num_actors
        self.minibatch_size = self.config["minibatch_size"]
        self.update_epochs = self.config["mini_epochs"]
        self.num_minibatches = self.batch_size // self.minibatch_size
        assert self.batch_size % self.minibatch_size == 0

        self._amp_batch_size = int(self.config["amp_batch_size"])
        self._amp_minibatch_size = int(self.config["amp_minibatch_size"])
        assert self._amp_minibatch_size <= self.minibatch_size

        self.obs = None
        self.last_lr = float(self.config["learning_rate"])
        self.frame = 0
        self.update_time = 0
        self.play_time = 0
        self.epoch_num = 0
        self.curr_frames = 0

        self.optimizer = optim.Adam(self.model.parameters(), self.last_lr, eps=1e-08, weight_decay=0.0)

        self.dataset = AMPDataset(self.batch_size, self.minibatch_size, self.horizon_length)

        self.games_to_track = self.config.get("games_to_track", 100)
        self.game_rewards = AverageMeter(self.value_size, self.games_to_track).to(self.device)
        self.game_lengths = AverageMeter(1, self.games_to_track).to(self.device)

        # Check folders
        # allows us to specify a folder where all experiments will reside
        self.train_dir = self.config.get("train_dir", "runs")

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_name = self.config["name"] + datetime.now().strftime("_%m-%d-%H-%M-%S")
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, "nn")
        self.summaries_dir = os.path.join(self.experiment_dir, "summaries")

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.writer = SummaryWriter(self.summaries_dir)
        self.algo_observer.after_init(self)

    def train(self):
        self.init_tensors()
        total_time = 0
        self.frame = 0

        # NOTE: Set the early termination condition to 0.25 (the original value)
        self.task_env._termination_distances[:] = 0.25

        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        model_output_file = os.path.join(self.nn_dir, self.config["name"])

        # self._init_amp_demo_buf()
        buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self._amp_batch_size))
        for _ in range(num_batches):
            demos = self.task_env.fetch_amp_obs_demo(self._amp_batch_size)
            self._amp_obs_demo_buffer.store({"amp_obs": demos})

        # MATCH xcxc debug -- init (norlg)
        # print("obs", self.obs.sum())
        # print("amb obs", self._amp_obs_demo_buffer._data_buf["amp_obs"].sum())
        # print("amp dataset idx", self.dataset._idx_buf[0])

        while True:
            self.epoch_num += 1

            ### pre_epoch
            if (self.epoch_num + 1) % self.task_env.motion_resampling_interval == 0:
                self.task_env.resample_motions()

            # Freeze running mean/std, so that the actor does not use the updated mean/std
            self.running_mean_std_temp = deepcopy(self.running_mean_std)
            self.running_mean_std_temp.freeze()

            ### Collect data
            self.set_eval()
            start_time = time.time()
            with torch.no_grad():
                batch_dict = self.collect_data()
            scaled_play_time = time.time() - start_time

            # Add amp obs
            demos = self.task_env.fetch_amp_obs_demo(self._amp_batch_size)
            self._amp_obs_demo_buffer.store({"amp_obs": demos})

            num_obs_samples = batch_dict["amp_obs"].shape[0]
            amp_obs_demo = self._amp_obs_demo_buffer.sample(num_obs_samples)["amp_obs"]
            batch_dict["amp_obs_demo"] = amp_obs_demo

            if self._amp_replay_buffer.get_total_count() == 0:
                amp_obs_replay = batch_dict["amp_obs"]
            else:
                amp_obs_replay = self._amp_replay_buffer.sample(num_obs_samples)["amp_obs"]
            batch_dict["amp_obs_replay"] = amp_obs_replay

            # MATCH xcxc debug -- amp obs buffers (norlg)
            # print("returns", batch_dict["returns"].sum())
            # print("amp_obs_demo", amp_obs_demo.sum())
            # print("amp_obs_replay", amp_obs_replay.sum())
            # print()

            ### Update the model
            update_time_start = time.time()
            train_info = None

            self.set_train()

            self.curr_frames = batch_dict.pop("played_frames")

            # self.prepare_dataset(batch_dict)
            returns = batch_dict["returns"]
            values = batch_dict["values"]
            advantages = torch.sum(returns - values, axis=1)
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # NOTE: prepare_dataset normalizes returns/values, which need to be updated
            # So these normalizers (self.value_mean_std) need to be in the training mode
            if self.normalize_value:
                self.value_mean_std.train()

                # CHECK ME: unnorm=False here. Why?
                values = self.value_mean_std(values)
                returns = self.value_mean_std(returns)

            dataset_dict = {
                "old_values": values,
                "old_logp_actions": batch_dict["neglogpacs"],
                "advantages": advantages,
                "returns": returns,
                "actions": batch_dict["actions"],
                "obs": batch_dict["obses"],
                "mu": batch_dict["mus"],
                "sigma": batch_dict["sigmas"],
                "amp_obs": batch_dict["amp_obs"],
                "amp_obs_demo": batch_dict["amp_obs_demo"],
                "amp_obs_replay": batch_dict["amp_obs_replay"],
            }
            self.dataset.update_values_dict(dataset_dict)

            # xcxc debug -- prepare dataset (no rlg)
            # for k, v in dataset_dict.items():
            #     try:
            #         print(k, v.sum())
            #     except:
            #         pass

            for _ in range(0, self.update_epochs):
                for i in range(len(self.dataset)):
                    curr_train_info = self.update_minibatch(self.dataset[i])

                    if train_info is None:
                        train_info = dict()
                        for k, v in curr_train_info.items():
                            train_info[k] = [v]
                    else:
                        for k, v in curr_train_info.items():
                            train_info[k].append(v)

            train_info["play_time"] = scaled_play_time
            train_info["update_time"] = time.time() - update_time_start
            train_info["disc_rewards"] = batch_dict["disc_rewards"]
            train_info["reward_raw"] = batch_dict["reward_raw"]
            train_info["mb_rewards"] = batch_dict["mb_rewards"]
            train_info["returns"] = batch_dict["returns"]

            # self._store_replay_amp_obs(batch_dict["amp_obs"])
            amp_obs = batch_dict["amp_obs"]
            buf_size = self._amp_replay_buffer.get_buffer_size()
            buf_total_count = self._amp_replay_buffer.get_total_count()
            if buf_total_count > buf_size:
                keep_probs = torch.tensor(
                    np.array([self._amp_replay_keep_prob] * amp_obs.shape[0]), dtype=torch.float, device=self.device
                )
                keep_mask = torch.bernoulli(keep_probs) == 1.0
                amp_obs = amp_obs[keep_mask]

            if amp_obs.shape[0] > buf_size:
                rand_idx = torch.randperm(amp_obs.shape[0])
                rand_idx = rand_idx[:buf_size]
                amp_obs = amp_obs[rand_idx]

            self._amp_replay_buffer.store({"amp_obs": amp_obs})

            ### Post_epoch
            # # CHECK ME: Is this needed?
            # # Freeze running mean/std, so that the actor does not use the updated mean/std
            # self.running_mean_std_temp = deepcopy(self.running_mean_std)  # Unfreeze running mean/std
            # self.running_mean_std_temp.freeze()

            # MATCH xcxc debug -- train epoch (norlg)
            #for k in ["kl", "entropy", "actor_loss", "critic_loss", "b_loss", "disc_loss", "disc_agent_logit", "disc_rewards"]:
            for k in ["kl"]:
                if isinstance(train_info[k], list):
                    print(k, torch.stack(train_info[k]).sum())
                else:
                    print(k, train_info[k].sum())

            ### Log the stats
            sum_time = time.time() - start_time
            total_time += sum_time
            scaled_time = sum_time
            curr_frames = self.curr_frames
            self.frame += curr_frames
            mean_rewards = self.game_rewards.get_mean()
            mean_lengths = self.game_lengths.get_mean()
            if self.print_stats:
                fps_step = curr_frames / scaled_play_time
                fps_total = curr_frames / scaled_time
                print(
                    f"epoch: {self.epoch_num}",
                    f"\trwd: {np.mean(mean_rewards):.1f}",
                    f"\tfps step: {fps_step:.1f}",
                    f"\tfps total: {fps_total:.1f}",
                    f"\teps len: {mean_lengths:.1f}",
                )

            train_info_dict = {
                "performance/total_fps": curr_frames / scaled_time,
                "performance/step_fps": curr_frames / scaled_play_time,
                "performance/update_time": train_info["update_time"],
                "performance/play_time": train_info["play_time"],
                "learning_rate/last_lr": train_info["last_lr"][-1] * train_info["lr_mul"][-1],
                "learning_rate/lr_mul": train_info["lr_mul"][-1],
                "learning_rate/e_clip": self.e_clip * train_info["lr_mul"][-1],
            }

            if "actor_loss" in train_info:
                train_info_dict.update(
                    {
                        "loss/actor_loss": mean_list(train_info["actor_loss"]).item(),
                        "loss/critic_loss": mean_list(train_info["critic_loss"]).item(),
                        "loss/bounds_loss": mean_list(train_info["b_loss"]).item(),
                        "loss/entropy": mean_list(train_info["entropy"]).item(),
                        "loss/clip_frac": mean_list(train_info["actor_clip_frac"]).item(),
                        "loss/kl": mean_list(train_info["kl"]).item(),
                    }
                )

            if "disc_loss" in train_info:
                disc_reward_std, disc_reward_mean = torch.std_mean(train_info["disc_rewards"])
                train_info_dict.update(
                    {
                        "disc/loss": mean_list(train_info["disc_loss"]).item(),
                        "disc/agent_acc": mean_list(train_info["disc_agent_acc"]).item(),
                        "disc/demo_acc": mean_list(train_info["disc_demo_acc"]).item(),
                        "disc/agent_logit": mean_list(train_info["disc_agent_logit"]).item(),
                        "disc/demo_logit": mean_list(train_info["disc_demo_logit"]).item(),
                        "disc/grad_penalty": mean_list(train_info["disc_grad_penalty"]).item(),
                        "disc/logit_loss": mean_list(train_info["disc_logit_loss"]).item(),
                        "disc/reward_mean": disc_reward_mean.item(),
                        "disc/reward_std": disc_reward_std.item(),
                    }
                )

            if "returns" in train_info:
                train_info_dict["rewards/returns"] = train_info["returns"].mean().item()

            if "mb_rewards" in train_info:
                train_info_dict["rewards/mb_rewards"] = train_info["mb_rewards"].mean().item()

            if "reward_raw" in train_info:
                reward_raw = train_info["reward_raw"].cpu().numpy().tolist()
                train_info_dict["rewards/body_pos"] = reward_raw[0]
                train_info_dict["rewards/body_rot"] = reward_raw[1]
                train_info_dict["rewards/lin_vel"] = reward_raw[2]
                train_info_dict["rewards/ang_vel"] = reward_raw[3]
                train_info_dict["rewards/power"] = reward_raw[4]

            for k, v in train_info_dict.items():
                self.writer.add_scalar(k, v, self.epoch_num)

            if wandb.run is not None:
                wandb.log(train_info_dict, step=self.epoch_num)

            # self.algo_observer.after_print_stats(frame, epoch_num, total_time)

            # save the checkpoint
            if self.save_freq > 0:
                if self.epoch_num % self.save_freq == 0:
                    self.save(model_output_file)

                    # save the intermediate checkpoints
                    int_model_output_file = model_output_file + "_" + str(self.epoch_num).zfill(8)
                    shutil.copyfile(model_output_file, int_model_output_file)

                    self.evaluate_model()

            if self.epoch_num > self.max_epochs:
                self.save(model_output_file)
                print("Reached the maximum number of epochs. Finshed training.")
                return

    def init_tensors(self):
        algo_info = {
            "num_agents": self.num_agents,
            "num_actors": self.num_actors,
            "horizon_length": self.horizon_length,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "value_size": 1,
        }
        self.experience_buffer = ExperienceBuffer(algo_info, self.device)
        self.experience_buffer.tensor_dict["next_obses"] = torch.zeros_like(self.experience_buffer.tensor_dict["obses"])
        self.experience_buffer.tensor_dict["next_values"] = torch.zeros_like(
            self.experience_buffer.tensor_dict["values"]
        )

        batch_size = self.num_agents * self.num_actors
        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)

        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict["amp_obs"] = torch.zeros(
            batch_shape + self.amp_obs_shape, device=self.device
        )

        amp_obs_demo_buffer_size = int(self.config["amp_obs_demo_buffer_size"])
        self._amp_obs_demo_buffer = ReplayBuffer(amp_obs_demo_buffer_size, self.device)

        self._amp_replay_keep_prob = self.config["amp_replay_keep_prob"]
        replay_buffer_size = int(self.config["amp_replay_buffer_size"])
        self._amp_replay_buffer = ReplayBuffer(replay_buffer_size, self.device)

        self.update_list = ["actions", "neglogpacs", "values", "mus", "sigmas"]
        self.tensor_list = self.update_list + ["obses", "dones", "next_obses", "amp_obs"]

    #####################################################################

    def collect_data(self):
        self.set_eval()

        done_indices = []
        reward_raw = None

        for n in range(self.horizon_length):
            # NOTE: done_indices = None resets all envs. done_indices = [] does not reset envs.
            self.obs = self.env_reset(done_indices)
            self.experience_buffer.update_data("obses", n, self.obs)

            res_dict = self.get_action_values(self.obs)
            for k in self.update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            # MATCH xcxc debug -- play steps, get_action_values (norlg)
            # print(n, ", len dones", len(done_indices))
            # print("obs", self.obs.sum())
            # print("actions", res_dict["actions"].sum())

            """Stepping the environment"""
            self.obs, rewards, self.dones, infos = self.env.step(res_dict["actions"])
            # print("new_obs", self.obs.sum())
            # print("rewards", rewards.sum())
            # print("amp obs", infos["amp_obs"].sum())

            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)

            # No special reward shaping used. Remove.
            # shaped_rewards = self.rewards_shaper(rewards)
            # shaped_rewards = rewards  # shape error

            self.experience_buffer.update_data("rewards", n, rewards)
            self.experience_buffer.update_data("next_obses", n, self.obs)
            self.experience_buffer.update_data("dones", n, self.dones)
            self.experience_buffer.update_data("amp_obs", n, infos["amp_obs"])

            reward_raw_mean = infos["reward_raw"].mean(dim=0)
            if reward_raw is None:
                reward_raw = reward_raw_mean
            else:
                reward_raw += reward_raw_mean

            terminated = infos["terminate"].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= 1.0 - terminated
            self.experience_buffer.update_data("next_values", n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[:: self.num_agents].flatten()

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            if True or self.task_env.viewer:
                self._amp_debug(infos)

        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_next_values = self.experience_buffer.tensor_dict["next_values"]

        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_amp_obs = self.experience_buffer.tensor_dict["amp_obs"]
        disc_rewards = self._calc_disc_rewards(mb_amp_obs)

        # Combine the task and disc rewards
        mb_rewards *= self._task_reward_w
        mb_rewards += self._disc_reward_w * disc_rewards

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict["played_frames"] = self.batch_size
        batch_dict["returns"] = swap_and_flatten01(mb_returns)
        batch_dict["mb_rewards"] = swap_and_flatten01(mb_rewards)
        batch_dict["reward_raw"] = reward_raw / self.horizon_length
        batch_dict["disc_rewards"] = swap_and_flatten01(disc_rewards)

        # MATCH xcxc debug -- play steps (norlg)
        # for k in ["amp_obs", "returns", "disc_rewards"]:
        #     print(k, batch_dict[k].sum())

        return batch_dict

    def get_action_values(self, obs_torch):
        processed_obs = self._preproc_obs(obs_torch)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": processed_obs,
        }

        # MATCH xcxc debug -- get_action_values (norlg)
        # print("preproc obs", processed_obs.sum())

        with torch.no_grad():
            res_dict = self.model(input_dict)

        if self.normalize_value:
            # CHECK ME: what's the difference between unnorm=True vs. False?
            res_dict["values"] = self.value_mean_std(res_dict["values"], unnorm=True)

        return res_dict

    def _eval_critic(self, obs_torch):
        # NOTE: conversion to dict is to keep rl-games compatibility
        processed_obs = {"obs": self._preproc_obs(obs_torch)}
        value = self.model.a2c_network.eval_critic(processed_obs)

        if self.normalize_value:
            # CHECK ME: what's the difference between unnorm=True vs. False?
            value = self.value_mean_std(value, unnorm=True)
        return value

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._disc_reward_scale
        return disc_r

    def discount_values(self, mb_fdones, mb_values, mb_rewards, mb_next_values):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            not_done = 1.0 - mb_fdones[t]
            not_done = not_done.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * mb_next_values[t] - mb_values[t]
            lastgaelam = delta + self.gamma * self.tau * not_done * lastgaelam
            mb_advs[t] = lastgaelam

        return mb_advs

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info["amp_obs"]
            amp_obs = amp_obs[0:1]
            disc_pred = self._eval_disc(amp_obs)
            disc_reward = self._calc_disc_rewards(amp_obs)

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            # print("disc_pred: ", disc_pred, disc_reward)

    #####################################################################

    def update_minibatch(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = input_dict["obs"]
        obs_batch_processed = self._preproc_obs(obs_batch, use_temp=self.temp_running_mean)

        amp_obs = input_dict["amp_obs"][0 : self._amp_minibatch_size]
        amp_obs = self._preproc_amp_obs(amp_obs)

        amp_obs_replay = input_dict["amp_obs_replay"][0 : self._amp_minibatch_size]
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)

        amp_obs_demo = input_dict["amp_obs_demo"][0 : self._amp_minibatch_size]
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch_processed,
            "amp_obs": amp_obs,
            "amp_obs_replay": amp_obs_replay,
            "amp_obs_demo": amp_obs_demo,
        }
        
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)

            # Calculate loss
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]
            disc_agent_logit = res_dict["disc_agent_logit"]
            disc_agent_replay_logit = res_dict["disc_agent_replay_logit"]
            disc_demo_logit = res_dict["disc_demo_logit"]

            # xcxc debug -- clip policy loss (no rlg)
            print("action_log_probs", action_log_probs.sum(), (action_log_probs**2).sum())
            print("advantage", advantage.sum(), (advantage**2).sum())

            a_info = self._clip_policy_loss(old_action_log_probs_batch, action_log_probs, advantage, self.e_clip)
            a_loss = a_info["actor_loss"]
            a_clipped = a_info["actor_clipped"].float()

            c_info = self._clip_value_loss(value_preds_batch, values, self.e_clip, return_batch, self.clip_value)
            c_loss = c_info["critic_loss"]

            b_loss = self.bound_loss(mu)

            a_loss = torch.mean(a_loss)
            a_clip_frac = torch.mean(a_clipped)
            c_loss = torch.mean(c_loss)
            b_loss = torch.mean(b_loss)
            entropy = torch.mean(entropy)

            disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
            disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, amp_obs_demo)
            disc_loss = disc_info["disc_loss"]

            loss = (
                a_loss
                + self.critic_coef * c_loss
                - self.entropy_coef * entropy
                + self.bounds_loss_coef * b_loss
                + self._disc_coef * disc_loss
            )

            a_info["actor_loss"] = a_loss
            a_info["actor_clip_frac"] = a_clip_frac
            c_info["critic_loss"] = c_loss

            # MATCH xcxc debug -- loss calculation (no rlg)
            print("a loss", a_loss.sum())
            print("a_clip_frac", a_clip_frac.sum())
            # print("c loss", c_loss.sum())
            # print("b loss", b_loss.sum())

            self.optimizer.zero_grad(set_to_none=True)

        self.scaler.scale(loss).backward()

        # Print gradient stats before optimizer step
        actor_grad_norm = 0
        for p in self.model.a2c_network.parameters():
            if p.grad is not None:
                actor_grad_norm += p.grad.norm().item()
        print(f"Before clip grad norm: {actor_grad_norm}")

        if self.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Print gradient stats before optimizer step
        actor_grad_norm = 0
        for p in self.model.a2c_network.parameters():
            if p.grad is not None:
                actor_grad_norm += p.grad.norm().item()
        print(f"After clip grad norm: {actor_grad_norm}")

        # # Update the model, without the scaler
        # self.optimizer.zero_grad(set_to_none=True)
        # loss.backward()
        # if self.truncate_grads:
        #     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        # self.optimizer.step()

        with torch.no_grad():
            kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch)

        # xcxc debug -- loss backward (no rlg)
        print("loss backward", loss.sum())
        print()

        self.train_result = {
            "entropy": entropy,
            "kl": kl_dist,
            "last_lr": self.last_lr,
            "lr_mul": 1.0,
            "b_loss": b_loss,
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(disc_info)

        return self.train_result

    def _clip_policy_loss(self, old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip):
        # clipping the policy loss
        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)
        clipped = torch.abs(ratio - 1.0) > curr_e_clip
        return {"actor_loss": a_loss, "actor_clipped": clipped.detach()}

    def _clip_value_loss(self, value_preds_batch, values, curr_e_clip, return_batch, clip_value):
        # clipping the value loss
        if clip_value:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
            value_losses = (values - return_batch) ** 2
            value_losses_clipped = (value_pred_clipped - return_batch) ** 2
            c_loss = torch.max(value_losses, value_losses_clipped)
        else:
            c_loss = (return_batch - values) ** 2

        return {"critic_loss": c_loss}

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.0
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss

    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        """
        disc_agent_logit: replay and current episode logit (fake examples)
        disc_demo_logit: disc_demo_logit logit
        obs_demo: gradient penalty demo obs (real examples)
        """
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.model.a2c_network.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(
            disc_demo_logit,
            obs_demo,
            grad_outputs=torch.ones_like(disc_demo_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if self._disc_weight_decay != 0:
            disc_weights = self.model.a2c_network.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        disc_info = {
            "disc_loss": disc_loss,
            "disc_grad_penalty": disc_grad_penalty.detach(),
            "disc_logit_loss": disc_logit_loss.detach(),
            "disc_agent_acc": disc_agent_acc.detach(),
            "disc_demo_acc": disc_demo_acc.detach(),
            "disc_agent_logit": disc_agent_logit.detach(),
            "disc_demo_logit": disc_demo_logit.detach(),
        }
        return disc_info

    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss

    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    def evaluate_model(self):
        pass

    def save(self, file_path):
        print("=> saving checkpoint '{}'".format(file_path))
        state_dict = self.get_model_weights()

        # Training state
        state_dict["epoch"] = self.epoch_num
        state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["frame"] = self.frame

        # Save the checkpoint
        torch.save(state_dict, file_path)
