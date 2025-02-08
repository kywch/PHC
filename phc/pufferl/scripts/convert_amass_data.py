import glob
import argparse
import os.path as osp
from uuid import uuid4

import torch
import numpy as np
from scipy.spatial.transform import Rotation as sRot

import joblib
from tqdm import tqdm

from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES

from phc import PHC_ROOT, BODY_MODEL_DIR
from phc.pufferl.poselib_skeleton import SkeletonTree, SkeletonState

# NOTE: hardcoded 66
SELECT_DOF = 22 * 3  # 22 SMPL joints, without fingers x 3. Replace fingers with dummy hands
SMPL_JOINT_NUM = len(SMPL_BONE_ORDER_NAMES)

TARGET_FRAME_RATE = 30
UPRIGHT_START = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--path", type=str, default="/workspace/dataset/AMASS")
    parser.add_argument("--occulusion_file", type=str, default=f"{PHC_ROOT}/sample_data/amass_copycat_occlusion_v3.pkl")
    parser.add_argument(
        "--name_offset", type=int, default=-3, help="The offset of the dataset name in the full file path"
    )
    args = parser.parse_args()

    process_split = "train"
    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": UPRIGHT_START,
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "replace_feet": True,
        "masterfoot": False,
        "big_ankle": True,
        "freeze_hand": False,
        "box_body": False,
        "master_range": 50,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": "smpl",
    }

    smpl_local_robot = SMPL_Robot(robot_cfg, data_dir=BODY_MODEL_DIR)
    if not osp.isdir(args.path):
        print("Please specify AMASS data path")

    all_pkls = glob.glob(f"{args.path}/**/*.npz", recursive=True)
    amass_occlusion = joblib.load(args.occulusion_file)
    amass_full_motion_dict = {}
    amass_splits = {
        "valid": ["HumanEva", "MPI_HDM05", "SFU", "MPI_mosh"],
        "test": ["Transitions_mocap", "SSM_synced"],
        "train": [
            "CMU",
            "MPI_Limits",
            "TotalCapture",
            "KIT",
            "EKUT",
            "TCD_handMocap",
            "BMLhandball",
            "DanceDB",
            "ACCAD",
            "BMLmovi",
            "BioMotionLab_NTroje",
            "Eyes_Japan_Dataset",
            "DFaust_67",
        ],  # Adding ACCAD
    }
    process_set = amass_splits[process_split]
    length_acc = []
    for data_path in tqdm(all_pkls):
        bound = 0

        splits = data_path.split("/")[args.name_offset :]
        key_name_dump = "0-" + "_".join(splits).replace(".npz", "")

        if splits[0] not in process_set:
            continue

        if key_name_dump in amass_occlusion:
            issue = amass_occlusion[key_name_dump]["issue"]
            if (issue == "sitting" or issue == "airborne") and "idxes" in amass_occlusion[key_name_dump]:
                bound = amass_occlusion[key_name_dump]["idxes"][0]  # This bounded is calculated assuming 30 FPS...
                if bound < 10:
                    print("bound too small", key_name_dump, bound)
                    continue
            else:
                print("issue irrecoverable", key_name_dump, issue)
                continue

        entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

        if "mocap_framerate" not in entry_data:
            continue
        framerate = entry_data["mocap_framerate"]

        if "0-KIT_442_PizzaDelivery02_poses" == key_name_dump:
            bound = -2

        skip = int(framerate / TARGET_FRAME_RATE)
        root_trans = entry_data["trans"][::skip, :]

        pose_aa = np.concatenate(
            [entry_data["poses"][::skip, :SELECT_DOF], np.zeros((root_trans.shape[0], 6))], axis=-1
        )

        betas = entry_data["betas"]
        gender = entry_data["gender"]
        num_frames = pose_aa.shape[0]

        if bound == 0:
            bound = num_frames

        root_trans = root_trans[:bound]
        pose_aa = pose_aa[:bound]
        num_frames = pose_aa.shape[0]
        if num_frames < 10:
            continue

        smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
        pose_aa_mj = pose_aa.reshape(num_frames, SMPL_JOINT_NUM, 3)[:, smpl_2_mujoco]
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(num_frames, SMPL_JOINT_NUM, 4)

        beta = np.zeros((16))
        gender_number, beta[:], gender = [0], 0, "neutral"
        # print("using neutral model")
        smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
        tmp_asset_file = f"/tmp/smpl/{uuid4()}.xml"
        smpl_local_robot.write_xml(tmp_asset_file)
        skeleton_tree = SkeletonTree.from_mjcf(tmp_asset_file)
        root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True,
        )

        if UPRIGHT_START:
            pose_quat_global = (
                (
                    sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy())
                    * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
                )
                .as_quat()
                .reshape(num_frames, -1, 4)
            )  # should fix pose_quat as well here...

            new_sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False
            )
            pose_quat = new_sk_state.local_rotation.numpy()

        pose_quat_global = new_sk_state.global_rotation.numpy()
        pose_quat = new_sk_state.local_rotation.numpy()

        new_motion_out = {}
        new_motion_out["pose_quat_global"] = pose_quat_global
        new_motion_out["pose_quat"] = pose_quat
        new_motion_out["trans_orig"] = root_trans
        new_motion_out["root_trans_offset"] = root_trans_offset
        new_motion_out["beta"] = beta
        new_motion_out["gender"] = gender
        new_motion_out["pose_aa"] = pose_aa
        new_motion_out["fps"] = TARGET_FRAME_RATE

        amass_full_motion_dict[key_name_dump] = new_motion_out
        # print(f"Processed {key_name_dump}")

    print("Processed", len(amass_full_motion_dict), "motions, saving to", args.path)

    if UPRIGHT_START:
        joblib.dump(amass_full_motion_dict, f"{args.path}/amass_train_upright.pkl", compress=True)
    else:
        joblib.dump(amass_full_motion_dict, f"{args.path}/amass_train.pkl", compress=True)
