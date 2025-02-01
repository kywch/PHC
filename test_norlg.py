import os
import os.path as osp

import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict

import isaacgym

from rl_games.common import env_configurations

from phc import flags
from phc.utils.config import set_np_formatting, set_seed

from debug_prim import Runner, create_rlgpu_env

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

    # cfg, cfg_train, logdir = load_cfg(args)
    flags.debug, flags.follow, flags.fixed, flags.divide_group, flags.no_collision_check, flags.fixed_path, flags.real_path,  flags.show_traj, flags.server_mode, flags.slow, flags.real_traj, flags.im_eval, flags.no_virtual_display, flags.render_o3d = \
        cfg.debug, cfg.follow, False, False, False, False, False, True, cfg.server_mode, False, False, cfg.im_eval, cfg.no_virtual_display, cfg.render_o3d

    flags.test = cfg.test
    flags.add_proj = cfg.add_proj
    flags.has_eval = cfg.has_eval
    flags.trigger_input = False

    cfg.train = not cfg.test
    set_seed(cfg.get("seed", -1), cfg.get("torch_deterministic", False))

    # Create default directories for weights and statistics
    cfg_train = cfg.learning
    cfg_train['params']['config']['network_path'] = cfg.output_path
    cfg_train['params']['config']['train_dir'] = cfg.output_path
    cfg_train["params"]["config"]["num_actors"] = cfg.env.num_envs
    
    if cfg.epoch > 0:
        cfg_train["params"]["load_checkpoint"] = True
        cfg_train["params"]["load_path"] = osp.join(cfg.output_path, cfg_train["params"]["config"]['name'] + "_" + str(cfg.epoch).zfill(8) + '.pth')
    elif cfg.epoch == -1:
        path = osp.join(cfg.output_path, cfg_train["params"]["config"]['name'] + '.pth')
        if osp.exists(path):
            cfg_train["params"]["load_path"] = path
            cfg_train["params"]["load_checkpoint"] = True
        else:
            print(path)
            raise Exception("no file to resume!!!!")

    os.makedirs(cfg.output_path, exist_ok=True)

    env_creator = lambda **kwargs: create_rlgpu_env(cfg, cfg_train, **kwargs)
    env_configurations.register("rlgpu", {"env_creator": env_creator, "vecenv_type": "RLGPU"})

    if not cfg.test:
        log_wandb = input("Do you want to log to wandb? [y/n] ")
        if log_wandb == "y":
            import wandb
            wandb.init(project="PHC")
            wandb.config.update(cfg, allow_val_change=True)
            wandb.run.name = cfg.exp_name
            wandb.run.save()

    runner = Runner(env_creator)
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
    "env.motion_file=sample_data/amass_train_take6_upright.pkl",
]

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend(TRAIN_SINGLE_PRIM)

    main()
