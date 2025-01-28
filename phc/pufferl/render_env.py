from isaacgym import gymapi
import gymtorch

import torch
import numpy as np

# TODO: remove flags
from phc.utils.flags import flags

from phc import PHC_ROOT
from phc.pufferl.humanoid_phc import HumanoidPHC
from phc.pufferl.torch_utils import to_torch


class HumanoidRenderEnv(HumanoidPHC):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        if not self.headless or flags.server_mode:
            self._build_marker_state_tensors()

    def pause_func(self, action):
        self.paused = not self.paused

    def next_func(self, action):
        self.resample_motions()

    def reset_func(self, action):
        self.reset()

    def record_func(self, action):
        self.recording = not self.recording
        self.recording_state_change_o3d = True
        self.recording_state_change_o3d_img = True
        self.recording_state_change = True  # only intialize from o3d.

    def hide_ref(self, action):
        flags.show_traj = not flags.show_traj

    # NOTE: check the arg i
    def render(self, sync_frame_time=False, i=0):
        super().render(sync_frame_time=sync_frame_time)

        if self.viewer or flags.server_mode:
            self._update_marker()

    def _create_envs(self, num_envs, spacing, num_per_row):
        if not self.headless or flags.server_mode:
            self._marker_handles = [[] for _ in range(num_envs)]
            self._load_marker_asset()

        if flags.add_proj:
            self._proj_handles = []
            self._load_proj_asset()

        super()._create_envs(num_envs, spacing, num_per_row)

    def _load_marker_asset(self):
        asset_root = str(PHC_ROOT / "phc/data/assets/urdf/")

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 0.0
        asset_options.density = 0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(
            self.sim, asset_root, "traj_marker.urdf", asset_options
        )
        self._marker_asset_small = self.gym.load_asset(
            self.sim, asset_root, "traj_marker_small.urdf", asset_options
        )

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        if not self.headless or flags.server_mode:
            self._build_marker(env_id, env_ptr)

        if flags.add_proj:
            self._build_proj(env_id, env_ptr)

    def _build_marker(self, env_id, env_ptr):
        default_pose = gymapi.Transform()
        for i in range(self._num_joints):
            marker_handle = self.gym.create_actor(
                env_ptr, self._marker_asset, default_pose, "marker", self.num_envs + 10, 1, 0
            )

            if i in self._track_bodies_id:
                self.gym.set_rigid_body_color(
                    env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0)
                )
            else:
                self.gym.set_rigid_body_color(
                    env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 1.0, 1.0)
                )
            self._marker_handles[env_id].append(marker_handle)

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._marker_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1]
        )[..., 1 : (1 + self._num_joints), :]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_rotation = self._marker_states[..., 3:7]

        self._marker_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(
            self._marker_handles, dtype=torch.int32, device=self.device
        )
        self._marker_actor_ids = self._marker_actor_ids.flatten()

    def _update_marker(self):
        if flags.show_traj:
            motion_times = (
                (self.progress_buf + 1) * self.dt
                + self._motion_start_times
                + self._motion_start_times_offset
            )  # + 1 for target.
            motion_res = self._get_state_from_motionlib_cache(
                self._sampled_motion_ids, motion_times, self._global_offset
            )
            (
                root_pos,
                root_rot,
                dof_pos,
                root_vel,
                root_ang_vel,
                dof_vel,
                smpl_params,
                limb_weights,
                pose_aa,
                ref_rb_pos,
                ref_rb_rot,
                ref_body_vel,
                ref_body_ang_vel,
            ) = (
                motion_res["root_pos"],
                motion_res["root_rot"],
                motion_res["dof_pos"],
                motion_res["root_vel"],
                motion_res["root_ang_vel"],
                motion_res["dof_vel"],
                motion_res["motion_bodies"],
                motion_res["motion_limb_weights"],
                motion_res["motion_aa"],
                motion_res["rg_pos"],
                motion_res["rb_rot"],
                motion_res["body_vel"],
                motion_res["body_ang_vel"],
            )

            self._marker_pos[:] = ref_rb_pos
            # self._marker_rotation[..., self._track_bodies_id, :] = ref_rb_rot[..., self._track_bodies_id, :]

            ## Only update the tracking points.
            if flags.real_traj:
                self._marker_pos[:] = 1000

            self._marker_pos[..., self._track_bodies_id, :] = ref_rb_pos[
                ..., self._track_bodies_id, :
            ]

        else:
            self._marker_pos[:] = 1000

        # ######### Heading debug #######
        # points = self.init_root_points()
        # base_quat = self._rigid_body_rot[0, 0:1]
        # base_quat = remove_base_rot(base_quat)
        # heading_rot = calc_heading_quat(base_quat)
        # show_points = quat_apply(heading_rot.repeat(1, points.shape[0]).reshape(-1, 4), points) + (self._rigid_body_pos[0, 0:1]).unsqueeze(1)
        # self._marker_pos[:] = show_points[:, :self._marker_pos.shape[1]]
        # ######### Heading debug #######

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(self._marker_actor_ids),
            len(self._marker_actor_ids),
        )
        return

    # NOTE: Used in "Heading debug" above
    def init_root_points(self):
        # For debugging purpose
        y = torch.tensor(np.linspace(-0.5, 0.5, 5), device=self.device, requires_grad=False)
        x = torch.tensor(np.linspace(0, 1, 5), device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_root_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs, self.num_root_points, 3, device=self.device, requires_grad=False
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

