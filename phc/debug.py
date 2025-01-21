import os
import os.path as osp

import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict

from isaacgym import gymapi
from isaacgym import gymutil

from rl_games.common import env_configurations

from phc.utils.flags import flags
from phc.utils.config import set_np_formatting, set_seed
from phc.utils.parse_task import parse_task

from run_hydra import build_alg_runner, RLGPUAlgoObserver, parse_sim_params


env_configurations.register('rlgpu', {'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs), 'vecenv_type': 'RLGPU'})

def create_rlgpu_env(**kwargs):
    # use_horovod = cfg_train['params']['config'].get('multi_gpu', False)
    # if use_horovod:
    #     import horovod.torch as hvd

    #     rank = hvd.rank()
    #     print("Horovod rank: ", rank)

    #     cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank

    #     args.device = 'cuda'
    #     args.device_id = rank
    #     args.rl_device = 'cuda:' + str(rank)

    #     cfg['rank'] = rank
    #     cfg['rl_device'] = 'cuda:' + str(rank)
    
    sim_params = parse_sim_params(cfg)
    args = EasyDict({
        "task": cfg.env.task, 
        "device_id": cfg.device_id,
        "rl_device": cfg.rl_device,
        "physics_engine": gymapi.SIM_PHYSX if not cfg.sim.use_flex else gymapi.SIM_FLEX,
        "headless": cfg.headless,
        "device": cfg.device,
    }) #### ZL: patch 
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    print(env.num_envs)
    print(env.num_actions)
    print(env.num_obs)
    print(env.num_states)

    return env



@hydra.main(
    version_base=None,
    config_path="../phc/data/cfg",
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
    
    algo_observer = RLGPUAlgoObserver()
    runner = build_alg_runner(algo_observer)
    runner.load(cfg_train)
    runner.reset()
    runner.run(cfg)

    print()

# PHC+: keypoint model, can getup from the ground and walk back
PHC_PLUS_KP = [
    "learning=im_mcp_big",
    "learning.params.network.ending_act=False",
    "exp_name=phc_comp_kp_2",
    "env.obs_v=7",
    "env=env_im_getup_mcp",
    "robot=smpl_humanoid",
    "robot.real_weight_porpotion_boxes=False",
    "env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl",
    "env.models=['output/HumanoidIm/phc_kp_2/Humanoid.pth']",
    "env.num_prim=3",
    "env.num_envs=1",
    "headless=False",
    "epoch=-1",
    "test=True",
]

# keypoint model
KP_MCP_FULL = [
    "learning=im_mcp",
    "exp_name=phc_kp_mcp_iccv",
    "env=env_im_getup_mcp",
    "robot=smpl_humanoid",
    "robot.freeze_hand=True",
    "robot.box_body=False",
    "env.z_activation=relu",
    "env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl",
    "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']",
    "env.num_envs=1",
    "env.obs_v=7",
    "headless=False",
    "epoch=-1",
    "test=True",
]

# AMASS eval, rot + keypoint model
AMASS_EVAL_KP = [
    "learning=im_mcp_big",
    "exp_name=phc_comp_3",
    "env=env_im_getup_mcp",
    "robot=smpl_humanoid",
    "env.zero_out_far=False",
    "robot.real_weight_porpotion_boxes=False",
    "env.num_prim=3",
    "env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl",
    "env.models=['output/HumanoidIm/phc_3/Humanoid.pth']",
    "env.num_envs=1024",
    "headless=False",
    "im_eval=True",
]



if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend(AMASS_EVAL_KP)

    main()


