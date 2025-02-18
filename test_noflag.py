import os
import copy
import random
import os.path as osp

import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
import isaacgym

import torch
import numpy as np

from phc.norlg_learning.env import create_rlgpu_env
from phc.norlg_learning.utils import DefaultAlgoObserver
from phc.norlg_learning.phc_agent import PHCAgent
from phc.norlg_learning.network import AMPBuilder, ModelAMPContinuous

RUN_EVAL = False


def set_np_formatting():
    np.set_printoptions(
        edgeitems=30,
        infstr="inf",
        linewidth=4000,
        nanstr="nan",
        precision=2,
        suppress=False,
        threshold=10000,
        formatter=None,
    )


def seed_everything(seed, strict=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if strict:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)  # raises runtime error if not deterministic
        # torch.set_deterministic_debug_mode("warn")  # prints out warnings if not deterministic


# Replace rlgames' torch_runner and factories
class Runner:
    def __init__(self, env_creator, algo_observer=None):
        self.env_creator = env_creator
        self.algo_observer = algo_observer

    def load(self, yaml_conf):
        self.default_config = yaml_conf["params"]
        self.load_config(copy.deepcopy(self.default_config))

    def load_config(self, params):  # params = cfg_train
        self.algo_params = params["algo"]
        self.algo_name = self.algo_params["name"]
        self.load_check_point = params["load_checkpoint"]
        self.exp_config = None

        self.seed = params.get("seed", None)
        seed_everything(self.seed, strict=params["config"]["device"] == "cpu")

        if self.load_check_point:
            print("Found checkpoint")
            print(params["load_path"])
            self.load_path = params["load_path"]

        # self.model is actually model builder, not the model itself
        self.model = self.make_model_builder(params)
        self.config = copy.deepcopy(params["config"])
        self.config["network"] = self.model

    def make_model_builder(self, params):
        network_builder = AMPBuilder()
        network_builder.load(params["network"])
        return ModelAMPContinuous(network_builder)

    def run(self, cfg):
        if "checkpoint" in cfg and cfg["checkpoint"] is not None:
            if len(cfg["checkpoint"]) > 0:
                self.load_path = cfg["checkpoint"]

        self.config["algo_observer"] = DefaultAlgoObserver()

        agent = PHCAgent(self.config, self.env_creator)

        if cfg.train:
            agent.config_train()
            agent.train()

        elif cfg.test:
            if self.load_path != "Base":
                agent.restore(self.load_path)
            agent.task_env.toggle_eval_mode()  # fixes the motion heading
            agent.play()

        else:
            raise ValueError("Unknown mode.")


@hydra.main(
    version_base=None,
    config_path="phc/data/cfg",
    config_name="config",
)
def main(cfg_hydra: DictConfig) -> None:
    global cfg_train
    global cfg

    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    cfg.train = not cfg.test

    # Create default directories for weights and statistics
    cfg_train = cfg.learning
    cfg_train["params"]["config"]["network_path"] = cfg.output_path
    cfg_train["params"]["config"]["train_dir"] = cfg.output_path
    cfg_train["params"]["config"]["num_actors"] = cfg.env.num_envs

    if cfg.get("device", None):
        cfg_train["device"] = cfg.device
        cfg_train["params"]["config"]["device"] = cfg.device

    if cfg.test and osp.exists(cfg.load_checkpoint):
        cfg_train["params"]["load_checkpoint"] = True
        cfg_train["params"]["load_path"] = cfg.load_checkpoint

    elif cfg.epoch > 0:
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

    if not cfg.test:
        log_wandb = input("Do you want to log to wandb? [y/n] ")
        if log_wandb == "y":
            import wandb

            wandb.init(project="PHC")
            wandb.config.update(cfg, allow_val_change=True)
            wandb.run.name = cfg.exp_name
            wandb.run.save()

    run_rlg = cfg.get("run_rlg", False)

    env_creator = lambda **kwargs: create_rlgpu_env(cfg, **kwargs)
    runner = Runner(env_creator, run_rlg)
    runner.load(cfg_train)
    runner.run(cfg)

    if not cfg.test and log_wandb == "y":
        wandb.finish()


TRAIN_SINGLE_PRIM = [
    "learning=im_big",
    "exp_name=phc_prim",
    "env=env_im",
    "robot=smpl_humanoid",
    # "env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl",
    # "env.motion_file=sample_data/amass_train_take6_upright.pkl",
    # "env.motion_file=sample_data/amass_train_11k_upright.pkl",
    "env.motion_file=sample_data/totalcapture_acting_poses.pkl",
    # "env.motion_file=sample_data/dfaust_one_leg_jump.pkl",
    # "env.num_envs=32",
    # "learning.params.config.horizon_length=32",
    # "learning.params.config.minibatch_size=1024",
    # "learning.params.config.amp_minibatch_size=1024",
    # "learning.params.config.horizon_length=4",
    # "learning.params.config.minibatch_size=128",
    # "learning.params.config.amp_minibatch_size=128",
    # "learning.params.config.save_frequency=100",
    # "device=cpu",
    # Ablate discriminator
    "learning.params.config.disc_coef=0",
    "learning.params.config.disc_reward_w=0",
]

EVALUATE_SINGLE_PRIM = [
    "learning=im_big",
    "exp_name=phc_prim",
    "env=env_im",
    "robot=smpl_humanoid",
    # "env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl",
    # "env.motion_file=sample_data/amass_train_take6_upright.pkl",
    # "env.motion_file=sample_data/acting_poses.pkl",
    "env.motion_file=sample_data/dfaust_one_leg_jump.pkl",
    "epoch=-1",
    "test=True",
    "headless=False",
    "env.num_envs=4",
    "load_checkpoint=hum2.pth",
]

if __name__ == "__main__":
    kwargs = EVALUATE_SINGLE_PRIM if RUN_EVAL else TRAIN_SINGLE_PRIM

    import sys

    if len(sys.argv) < 3:
        sys.argv.extend(kwargs)
    else:
        raise ValueError("Too many arguments")

    set_np_formatting()

    main()
