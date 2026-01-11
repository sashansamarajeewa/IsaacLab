# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
from typing import Optional
from isaaclab.sensors import CameraCfg
import torch

import carb
from pink.tasks import DampingTask, FrameTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1t2_retargeter import (
    GR1T2RetargeterCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from . import mdp

from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG  # isort: skip

ASSET_SCALE = (2.0, 2.0, 2.0)


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):

    # Table
    packing_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/assembly/packing_table/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # ObstacleFront
    obstacle_front = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleFront",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0, 0.67, 1.02]),
        spawn=UsdFileCfg(
            usd_path="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/assembly/obstacle/obstacle_front/obstacle_front.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # ObstacleLeft
    obstacle_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleLeft",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.37, 0.48, 1.02]),
        spawn=UsdFileCfg(
            usd_path="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/assembly/obstacle/obstacle_side/obstacle_side.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # ObstacleRight
    obstacle_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleRight",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.37, 0.48, 1.02]),
        spawn=UsdFileCfg(
            usd_path="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/assembly/obstacle/obstacle_side/obstacle_side.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # FrontLeftLeg
    desk_leg_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FrontLeftLeg",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.02, 0.44, 1.08], rot=[0.0, 0.0, -0.7071, 0.7071]
        ),
        spawn=UsdFileCfg(
            usd_path="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/assembly/desk/desk_leg_1.usd",
            scale=ASSET_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1),
        ),
    )

    # FrontRightLeg
    desk_leg_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FrontRightLeg",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.245, 0.5, 1.085]),
        spawn=UsdFileCfg(
            usd_path="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/assembly/desk/desk_leg_2.usd",
            scale=ASSET_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1),
        ),
    )

    # BackLeftLeg
    desk_leg_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BackLeftLeg",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.245, 0.59, 1.085]),
        spawn=UsdFileCfg(
            usd_path="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/assembly/desk/desk_leg_3.usd",
            scale=ASSET_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1),
        ),
    )
    
    # BackRightLeg
    desk_leg_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BackRightLeg",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.245, 0.59, 1.085]),
        spawn=UsdFileCfg(
            usd_path="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/assembly/desk/desk_leg_4.usd",
            scale=ASSET_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1),
        ),
    )
    
    # DeskTop
    desk_top = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DeskTop",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.245, 0.59, 1.085]),
        spawn=UsdFileCfg(
            usd_path="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/assembly/desk/desk_top.usd",
            scale=ASSET_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=5),
        ),
    )

    # object = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=[-0.6, 0.47, 0.9996], rot=[1, 0, 0, 0]
    #     ),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
    #         scale=(0.75, 0.75, 0.75),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #     ),
    # )

    head_camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/GR1T2_fourier_hand_6dof/head_yaw_link/HeadCamera",
        height=720,
        width=1280,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=6.0),
        offset=CameraCfg.OffsetCfg(
            pos=(0.11, 0.0, 0.05),
            rot=(0.65328, 0.2706, -0.2706, -0.65328),
            convention="opengl",
        ),
    )

    # Humanoid robot configured for pick-place manipulation tasks
    robot: ArticulationCfg = GR1T2_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.93),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": -1.5708,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": -1.5708,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # --
                "head_.*": 0.0,
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                "R_.*": 0.0,
                "L_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
        ],
        hand_joint_names=[
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_thumb_proximal_yaw_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            "L_index_intermediate_joint",
            "L_middle_intermediate_joint",
            "L_pinky_intermediate_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_pitch_joint",
            "R_index_intermediate_joint",
            "R_middle_intermediate_joint",
            "R_pinky_intermediate_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_pitch_joint",
            "L_thumb_distal_joint",
            "R_thumb_distal_joint",
        ],
        target_eef_link_names={
            "left_wrist": "left_hand_pitch_link",
            "right_wrist": "right_hand_pitch_link",
        },
        # the robot in the sim scene we are controlling
        asset_name="robot",
        # Configuration for the IK controller
        # The frames names are the ones present in the URDF file
        # The urdf has to be generated from the USD that is being used in the scene
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="base_link",
            num_hand_joints=22,
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,  # Determines whether to pink solver will fail due to a joint limit violation
            variable_input_tasks=[
                FrameTask(
                    "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=12,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                FrameTask(
                    "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=12,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                DampingTask(
                    cost=0.5,  # [cost] * [s] / [rad]
                ),
                NullSpacePostureTask(
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
                        "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
                    ],
                    controlled_joints=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_pitch_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_pitch_joint",
                        # "waist_yaw_joint",
                        # "waist_pitch_joint",
                        # "waist_roll_joint",
                    ],
                ),
            ],
            fixed_input_tasks=[],
            xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(
            func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        robot_root_rot = ObsTerm(
            func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        # object_pos = ObsTerm(
        #     func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")}
        # )
        # object_rot = ObsTerm(
        #     func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")}
        # )
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(
            func=mdp.get_eef_pos, params={"link_name": "left_hand_roll_link"}
        )
        left_eef_quat = ObsTerm(
            func=mdp.get_eef_quat, params={"link_name": "left_hand_roll_link"}
        )
        right_eef_pos = ObsTerm(
            func=mdp.get_eef_pos, params={"link_name": "right_hand_roll_link"}
        )
        right_eef_quat = ObsTerm(
            func=mdp.get_eef_quat, params={"link_name": "right_hand_roll_link"}
        )

        hand_joint_state = ObsTerm(
            func=mdp.get_robot_joint_state, params={"joint_names": ["R_.*", "L_.*"]}
        )
        head_joint_state = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={
                "joint_names": ["head_pitch_joint", "head_roll_joint", "head_yaw_joint"]
            },
        )

        # object = ObsTerm(
        #     func=mdp.object_obs,
        #     params={
        #         "left_eef_link_name": "left_hand_roll_link",
        #         "right_eef_link_name": "right_hand_roll_link",
        #     },
        # )

        head_camera_rgb = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("head_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        head_camera_depth = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("head_camera"),
                "data_type": "distance_to_image_plane",
                "normalize": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # object_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum,
    #     params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")},
    # )

    # success = DoneTerm(
    #     func=mdp.task_done_pick_place, params={"task_link_name": "right_hand_roll_link"}
    # )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_object = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": [-0.01, 0.01],
    #             "y": [-0.01, 0.01],
    #         },
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object"),
    #     },
    # )


@configclass
class AssemblyDeskGR1T2EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    # OpenXR hand tracking has 26 joints per hand
    NUM_OPENXR_HAND_JOINTS = 26

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    # Idle action to hold robot in default pose
    # Action format: [left arm pos (3), left arm quat (4), right arm pos (3), right arm quat (4),
    #                 left hand joint pos (11), right hand joint pos (11)]
    idle_action = torch.tensor(
        [
            -0.22878,
            0.2536,
            1.0953,
            0.5,
            0.5,
            -0.5,
            0.5,
            0.22878,
            0.2536,
            1.0953,
            0.5,
            0.5,
            -0.5,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.7,
            friction_combine_mode="multiply",
        ),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            min_position_iteration_count=96,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,
        ),
        render=sim_utils.RenderCfg(enable_translucency=True),
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5  # 6
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 200  # 120Hz
        self.sim.render_interval = 2  # 6
        self.sim.physx.enable_ccd = False  # True
        carb.settings.get_settings().set_int("rtx/translucency/maxRefractionBounces", 2)

        # Convert USD to URDF and change revolute joints to fixed
        temp_urdf_output_path, temp_urdf_meshes_output_path = (
            ControllerUtils.convert_usd_to_urdf(
                self.scene.robot.spawn.usd_path,
                self.temp_urdf_dir,
                force_conversion=True,
            )
        )

        # Set the URDF and mesh paths for the IK controller
        self.actions.upper_body_ik.controller.urdf_path = temp_urdf_output_path
        self.actions.upper_body_ik.controller.mesh_path = temp_urdf_meshes_output_path

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        GR1T2RetargeterCfg(
                            enable_visualization=True,
                            # number of joints in both hands
                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "manusvive": ManusViveCfg(
                    retargeters=[
                        GR1T2RetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
