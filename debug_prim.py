import os
import os.path as osp
import copy
import random

import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
import isaacgym

import numpy as np
import torch

from rl_games.common import env_configurations, vecenv

from phc import flags
from phc.utils.config import set_np_formatting, set_seed

# RLG
import phc.learning.im_amp_players as rlg_players
import phc.learning.im_amp as rlg_agent

# NO RLG
from phc.norlg_learning.env import create_rlgpu_env
from phc.norlg_learning.utils import DefaultRewardsShaper, DefaultAlgoObserver
from phc.norlg_learning.network import AMPBuilder, ModelAMPContinuous
from phc.norlg_learning.phc_agent import PHCAgent

# TODO: Remove this
from phc.run_hydra import RLGPUEnv
vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

RUN_RLG = False
RUN_EVAL = False
WANDB_TRACK = False


# Replace rlgames' torch_runner and factories
class Runner:
    def __init__(self, env_creator, algo_observer=None):
        self.env_creator = env_creator
        self.algo_observer = algo_observer
        # torch.backends.cudnn.benchmark = True  # make non-deterministic

    def load(self, yaml_conf):
        self.default_config = yaml_conf["params"]
        self.load_config(copy.deepcopy(self.default_config))

        # if 'experiment_config' in yaml_conf:
        #     self.exp_config = yaml_conf['experiment_config']

    def load_config(self, params):  # params = cfg_train
        self.seed = params.get("seed", None)

        self.algo_params = params["algo"]
        self.algo_name = self.algo_params["name"]
        self.load_check_point = params["load_checkpoint"]
        self.exp_config = None

        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            os.environ["PYTHONHASHSEED"] = str(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

            if params["config"]["device"] == "cpu":
                # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)  # raises runtime error if not deterministic
                # torch.set_deterministic_debug_mode("warn")  # prints out warnings if not deterministic

        if self.load_check_point:
            print("Found checkpoint")
            print(params["load_path"])
            self.load_path = params["load_path"]

        # self.model is actually model builder, not the model itself
        self.model = self.make_model_builder(params)
        self.config = copy.deepcopy(params["config"])

        self.config["reward_shaper"] = DefaultRewardsShaper(**self.config["reward_shaper"])
        self.config["network"] = self.model

    def make_model_builder(self, params):
        network_builder = AMPBuilder()  # if not RUN_RLG else RLG_AMPBuilder()
        network_builder.load(params["network"])
        model_builder = ModelAMPContinuous(
            network_builder
        )  # if not RUN_RLG else RLG_ModelAMPContinuous(network_builder)
        return model_builder

    def run(self, args):
        if "checkpoint" in args and args["checkpoint"] is not None:
            if len(args["checkpoint"]) > 0:
                self.load_path = args["checkpoint"]

        if args["train"]:
            self.run_train()

        elif args["play"]:
            print("Started to play")
            player = self.create_player()
            if self.load_path != "Base":
                player.restore(self.load_path)
            player.play()

        else:
            raise ValueError(f"Unknown command: {args}")

    def create_player(self):
        if RUN_RLG:
            return rlg_players.IMAMPPlayerContinuous(self.config)
        else:
            return PHCAgent(self.config, self.env_creator)

    def run_train(self):
        if self.algo_observer is None:
            self.algo_observer = DefaultAlgoObserver()
        self.config["algo_observer"] = self.algo_observer

        if RUN_RLG:
            self.config["features"] = {"observer": self.algo_observer}
            agent = rlg_agent.IMAmpAgent(base_name="run", config=self.config)

        else:
            agent = PHCAgent(self.config, self.env_creator)
            agent.config_train()
            # vec_env = self.env_creator()
            # vec_env = RLGPUEnvWrapper(vec_env)
            # agent = ASEAgent(self.config, vec_env)

        if self.load_check_point and (self.load_path is not None):
            agent.restore(self.load_path)

        agent.train()

        # NOTE: If the training curves do not match, we may need to count the function calls
        # if PROFILE:
        #     # To count the function calls
        #     import cProfile
        #     import pstats
        #     from pstats import SortKey

        #     file_prefix = f"stats_{'rlg' if RUN_RLG else 'norlg'}.profile"
        #     def agent_train():
        #         agent.train()

        #     profiler = cProfile.Profile()
        #     profiler.runctx('agent_train()', globals(), locals())
        #     profiler.dump_stats(file_prefix + ".profile")

        #     with open(file_prefix + ".txt", "w") as f:
        #         p = pstats.Stats(file_prefix + ".profile", stream=f)
        #         p.sort_stats(SortKey.TIME).print_stats(200)

        # else:
        #     agent.train()


@hydra.main(
    version_base=None,
    config_path="phc/data/cfg",
    config_name="config",
)
def main(cfg_hydra: DictConfig) -> None:
    global cfg_train
    global cfg

    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))

    set_np_formatting()

    (
        flags.debug,
        flags.follow,
        flags.fixed,
        flags.divide_group,
        flags.no_collision_check,
        flags.fixed_path,
        flags.real_path,
        flags.show_traj,
        flags.server_mode,
        flags.slow,
        flags.real_traj,
        flags.im_eval,
        flags.no_virtual_display,
        flags.render_o3d,
    ) = (
        cfg.debug,
        cfg.follow,
        False,
        False,
        False,
        False,
        False,
        True,
        cfg.server_mode,
        False,
        False,
        cfg.im_eval,
        cfg.no_virtual_display,
        cfg.render_o3d,
    )

    flags.test = cfg.test
    flags.add_proj = cfg.add_proj
    flags.has_eval = cfg.has_eval
    flags.trigger_input = False

    cfg.train = not cfg.test
    set_seed(cfg.get("seed", -1), cfg.get("torch_deterministic", False))

    # Create default directories for weights and statistics
    cfg_train = cfg.learning
    cfg_train["params"]["config"]["device"] = cfg.device
    cfg_train["params"]["config"]["network_path"] = cfg.output_path
    cfg_train["params"]["config"]["train_dir"] = cfg.output_path
    cfg_train["params"]["config"]["num_actors"] = cfg.env.num_envs

    if cfg.epoch > 0:
        cfg_train["params"]["load_checkpoint"] = True
        cfg_train["params"]["load_path"] = osp.join(
            cfg.output_path, cfg_train["params"]["config"]["name"] + "_" + str(cfg.epoch).zfill(8) + ".pth"
        )
    elif cfg.epoch == -1:
        path = osp.join(cfg.output_path, cfg_train["params"]["config"]["name"] + ".pth")
        if osp.exists(path):
            cfg_train["params"]["load_path"] = path
            cfg_train["params"]["load_checkpoint"] = True
        else:
            print(path)
            raise Exception("no file to resume!!!!")

    os.makedirs(cfg.output_path, exist_ok=True)

    env_creator = lambda **kwargs: create_rlgpu_env(cfg, **kwargs)
    env_configurations.register("rlgpu", {"env_creator": env_creator, "vecenv_type": "RLGPU"})

    if WANDB_TRACK and not RUN_EVAL and not cfg.test:
        log_wandb = input("Do you want to log to wandb? [y/n]")
        if log_wandb == "y":
            import wandb

            wandb.init(project="PHC")
            wandb.config.update(cfg, allow_val_change=True)
            wandb.run.name = cfg.exp_name
            wandb.run.save()

    runner = Runner(env_creator)
    runner.load(cfg_train)
    runner.run(cfg)

    if WANDB_TRACK and not RUN_EVAL and not cfg.test:
        wandb.finish()


TRAIN_SINGLE_PRIM = [
    "learning=im_big",
    "exp_name=phc_prim",
    "env=env_im",
    "robot=smpl_humanoid",
    # "env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl",
    "env.motion_file=sample_data/amass_train_take6_upright.pkl",
    "env.num_envs=32",
    "learning.params.config.minibatch_size=1024",
    "learning.params.config.amp_minibatch_size=1024",
    # "learning.params.config.save_frequency=3",
    "device=cpu",
]

EVALUATE_SINGLE_PRIM = [
    "learning=im_big",
    "exp_name=phc_prim",
    "env=env_im",
    "robot=smpl_humanoid",
    # "env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl",
    "env.motion_file=sample_data/amass_train_take6_upright.pkl",
    "epoch=-1",
    "test=True",
    "headless=False",
    "env.num_envs=4",
]


if __name__ == "__main__":
    kwargs = EVALUATE_SINGLE_PRIM if RUN_EVAL else TRAIN_SINGLE_PRIM

    import sys

    if len(sys.argv) == 1:
        sys.argv.extend(kwargs)

    main()
