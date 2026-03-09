from .base import (
    BaseGuide,
    ang_deg,
    bind_base_white_for_moving_parts,
    first_descendant_with_rigid_body,
    resolve_env_scoped_path,
    spawn_ghost_preview,
    MaterialRegistry,
    update_ghost_preview_pose,
)
from pxr import UsdGeom, Usd, Gf
from typing import List, Optional, Tuple

# ------------------- Stable Guide -------------------


class StableGuide(BaseGuide):

    SEQUENCE = [
        "TableTop",
        "TableTop",
        "FrontRightLeg",
        "FrontLeftLeg",
        "TableTop",
        "BackRightLeg",
        "BackLeftLeg",
    ]
    MOVING_PARTS = (
        "TableTop",
        "FrontRightLeg",
        "FrontLeftLeg",
        "BackRightLeg",
        "BackLeftLeg",
    )
    STATIC_PARTS = ("ObstacleLeft", "ObstacleFront", "ObstacleRight")
    
    SCENE_KEY_MAP = {
        "TableTop": "square_top",
        "FrontRightLeg": "square_leg2",
        "FrontLeftLeg": "square_leg1",
        "BackRightLeg": "square_leg4",
        "BackLeftLeg": "square_leg3",
    }

    SNAP_PLAN = {
        1: ["TableTop"],
        2: ["TableTop", "FrontRightLeg"],
        3: ["TableTop", "FrontRightLeg", "FrontLeftLeg"],
        4: ["TableTop"],
        5: ["TableTop", "FrontRightLeg", "FrontLeftLeg", "BackRightLeg"],
        6: ["TableTop", "FrontRightLeg", "FrontLeftLeg", "BackRightLeg", "BackLeftLeg"],
    }

    tol_z_dbox_t = 1.07  # distance between table top and table origin along Z

    tgt_table_top_pos = Gf.Vec3d(
        0.1871977597475052, 0.4573464095592499, 1.0254216194152832
    )
    tgt_table_top_quat = Gf.Quatd(
        -0.00033001427073031664,
        Gf.Vec3d(-0.00033023953437805176, -0.7071070671081543, -0.7071064114570618),
    )
    tgt_front_right_leg_pos = Gf.Vec3d(
        0.2996618151664734, 0.34579798579216003, 1.119673728942871
    )
    tgt_front_right_leg_quat = Gf.Quatd(
        -0.03353802114725113,
        Gf.Vec3d(-0.0332774743437767, 0.7061654925346375, 0.7064688205718994),
    )
    tgt_front_left_leg_pos = Gf.Vec3d(
        0.07462520152330399, 0.34615617990493774, 1.119046688079834
    )
    tgt_front_left_leg_quat = Gf.Quatd(
        0.7062713503837585,
        Gf.Vec3d(0.7065197825431824, 0.03177633136510849, 0.031638968735933304),
    )
    tgt_stable_top_rot_pos = Gf.Vec3d(
        0.18748028576374054, 0.45748019218444824, 1.025420904159546
    )
    tgt_stable_top_rot_quat = Gf.Quatd(
        -0.7071067094802856,
        Gf.Vec3d(-0.7071069478988647, 4.2297080653952435e-05, 4.223044015816413e-05),
    )
    tgt_front_right_leg_rot_pos = Gf.Vec3d(
        0.07515661418437958, 0.5694549083709717, 1.1196768283843994
    )
    tgt_front_right_leg_rot_quat = Gf.Quatd(
        0.704781174659729,
        Gf.Vec3d(0.704741895198822, 0.057533618062734604, 0.05755211412906647),
    )
    tgt_front_left_leg_rot_pos = Gf.Vec3d(
        0.300271600484848, 0.5687900185585022, 1.1190509796142578
    )
    tgt_front_left_leg_rot_quat = Gf.Quatd(
        0.039292220026254654,
        Gf.Vec3d(0.03950197994709015, -0.706062912940979, -0.7059540748596191),
    )
    tgt_back_right_leg_pos = Gf.Vec3d(
        0.29954004287719727, 0.3438337445259094, 1.119034767150879
    )
    tgt_back_right_leg_quat = Gf.Quatd(
        0.7061378359794617,
        Gf.Vec3d(0.705807626247406, 0.039946526288986206, 0.04011766240000725),
    )
    tgt_back_left_leg_pos = Gf.Vec3d(
        0.07426538318395615, 0.3463539183139801, 1.1190378665924072
    )
    tgt_back_left_leg_quat = Gf.Quatd(
        0.7065831422805786,
        Gf.Vec3d(0.7067301869392395, 0.025240086019039154, 0.025214826688170433),
    )

    def __init__(self):
        super().__init__()
        self._checks = [
            self._check_pickup_table_top,
            self._check_braced_table_top,
            self._check_front_right_leg_insert,
            self._check_front_left_leg_insert,
            self._check_table_top_rotation,
            self._check_back_right_leg_insert,
            self._check_back_left_leg_insert,
        ]
        # Resolved prim paths. Moving parts - rigid body prim if available
        self._paths: dict[str, Optional[str]] = {}
        # Asset root paths for ghosts
        self._asset_roots: dict[str, Optional[str]] = {
            "TableTop": None,
            "FrontRightLeg": None,
            "FrontLeftLeg": None,
            "BackRightLeg": None,
            "BackLeftLeg": None,
        }
        # Cached static world poses for this episode
        self._static_table_pos: Optional[Gf.Vec3d] = None
        self._static_obstacles: dict[str, Optional[Tuple[Gf.Vec3d, Gf.Quatd]]] = {
            "ObstacleLeft": None,
            "ObstacleFront": None,
            "ObstacleRight": None,
        }

        # Target poses for ghost previews
        self._target_poses: dict[str, Optional[Tuple[Gf.Vec3d, Gf.Quatd]]] = {
            "TableTop": None,
            "FrontRightLeg": None,
            "FrontLeftLeg": None,
            "BackRightLeg": None,
            "BackLeftLeg": None,
        }

        # Ghost prim paths by logical name
        self._ghost_paths_by_name: dict[str, str] = {}

    # ------------------- reset -------------------

    def on_reset(self, env):
        super().on_reset(env)
        stage: Usd.Stage = env.scene.stage
        env_ns: str = env.scene.env_ns
        self._paths.clear()
        self._asset_roots = {
            "TableTop": None,
            "FrontRightLeg": None,
            "FrontLeftLeg": None,
            "BackRightLeg": None,
            "BackLeftLeg": None,
        }
        self._target_poses = {
            "TableTop": None,
            "FrontRightLeg": None,
            "FrontLeftLeg": None,
            "BackRightLeg": None,
            "BackLeftLeg": None,
        }
        self._ghost_paths_by_name.clear()
        self._static_table_pos = None
        self._static_obstacles = {
            "ObstacleLeft": None,
            "ObstacleFront": None,
            "ObstacleRight": None,
        }

        # Table (static)
        table_path = resolve_env_scoped_path(stage, env_ns, "PackingTable")
        self._paths["Table"] = table_path

        # Obstacles (static)
        for name in self.STATIC_PARTS:
            self._paths[name] = resolve_env_scoped_path(stage, env_ns, name)

        # Moving parts - rigid body prim if present else root
        for name in self.MOVING_PARTS:
            root_path = resolve_env_scoped_path(stage, env_ns, name)
            self._asset_roots[name] = root_path
            if not root_path:
                self._paths[name] = None
                continue
            rb_prim = first_descendant_with_rigid_body(
                stage, stage.GetPrimAtPath(root_path)
            )
            self._paths[name] = (
                str(rb_prim.GetPath()) if rb_prim and rb_prim.IsValid() else root_path
            )

        bind_base_white_for_moving_parts(stage, self.MOVING_PARTS)

        # Cache static world poses once
        cache = UsdGeom.XformCache()
        if self._paths.get("Table"):
            prim = stage.GetPrimAtPath(self._paths["Table"])
            if prim and prim.IsValid():
                self._static_table_pos = cache.GetLocalToWorldTransform(
                    prim
                ).ExtractTranslation()

        for name in self.STATIC_PARTS:
            p = self._paths.get(name)
            if not p:
                continue
            prim = stage.GetPrimAtPath(p)
            if prim and prim.IsValid():
                xf = cache.GetLocalToWorldTransform(prim)
                self._static_obstacles[name] = (
                    xf.ExtractTranslation(),
                    xf.ExtractRotation().GetQuat(),
                )

        # --------- Compute simple target poses for previews ---------
        if (
            self._static_table_pos is not None
            and self._static_obstacles["ObstacleLeft"] is not None
            and self._static_obstacles["ObstacleFront"] is not None
        ):

            # target TableTop braced in corner
            self._target_poses["TableTop"] = (
                self.tgt_table_top_pos,
                self.tgt_table_top_quat,
            )

            # target FrontRightLeg inserted to TableTop
            self._target_poses["FrontRightLeg"] = (
                self.tgt_front_right_leg_pos,
                self.tgt_front_right_leg_quat,
            )

            # target FrontLeftLeg inserted to TableTop
            self._target_poses["FrontLeftLeg"] = (
                self.tgt_front_left_leg_pos,
                self.tgt_front_left_leg_quat,
            )

            # target BackRightLeg inserted to TableTop
            self._target_poses["BackRightLeg"] = (
                self.tgt_back_right_leg_pos,
                self.tgt_back_right_leg_quat,
            )

            # target BackLeftLeg inserted to TableTop
            self._target_poses["BackLeftLeg"] = (
                self.tgt_back_left_leg_pos,
                self.tgt_back_left_leg_quat,
            )

        # --------- Spawn ghosts at target poses ---------
        stage = self._stage
        if stage is not None and getattr(self, "enable_ghosts", True):
            for name in self.MOVING_PARTS:
                root = self._asset_roots.get(name)
                tgt = self._target_poses.get(name)
                if not root or not tgt:
                    continue
                tgt_pos, tgt_quat = tgt
                ghost_root_path = f"{env_ns}/Ghosts/{name}_Ghost"
                ghost_path = spawn_ghost_preview(
                    stage=stage,
                    source_root_path=root,
                    target_pos=tgt_pos,
                    target_rot=tgt_quat,
                    ghost_root_path=ghost_root_path,
                    ghost_mat_path=MaterialRegistry.ghost_path,
                )
                self._ghost_paths_by_name[name] = ghost_path

        # Initialize ghost visibility to step 0
        if getattr(self, "enable_ghosts", True):
            self.update_ghost_visibility_for_step(0)

    def get_all_instructions(self) -> list[str]:
        total = len(self.SEQUENCE)
        base_steps = [
            f"Step 1/{total}: Pick up Table Top",
            f"Step 2/{total}: Brace Table Top against the front and right corner obstacles",
            f"Step 3/{total}: Insert Front Right Leg and screw clockwise until tight",
            f"Step 4/{total}: Insert Front Left Leg and screw clockwise until tight",
            f"Step 5/{total}: Rotate Table Top by 180°",
            f"Step 6/{total}: Insert Back Right Leg and screw clockwise until tight",
            f"Step 7/{total}: Insert Back Left Leg and screw clockwise until tight",
        ]
        base_steps.append("Assembly complete!")
        return base_steps

    # ---------------------- checks ----------------------

    def _check_pickup_table_top(self) -> bool:
        if self._static_table_pos is None:
            return False
        top_pose = self.get_live_part_pose("TableTop")
        if not top_pose:
            return False
        top_pos, _ = top_pose
        return (top_pos[2] - self._static_table_pos[2]) >= self.tol_z_dbox_t
        # return True

    def _check_braced_table_top(self) -> bool:
        tgt = self._target_poses.get("TableTop")
        live = self.get_live_part_pose("TableTop")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 and ang_err <= 3.5
        #return True

    def _check_front_right_leg_insert(self) -> bool:
        tgt = self._target_poses.get("FrontRightLeg")
        live = self.get_live_part_pose("FrontRightLeg")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 and ang_err <= 3.5
        #return True

    def _check_front_left_leg_insert(self) -> bool:
        tgt = self._target_poses.get("FrontLeftLeg")
        live = self.get_live_part_pose("FrontLeftLeg")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        result = pos_err <= 0.01 and ang_err <= 3.5
        if result:
            self._target_poses["TableTop"] = (
                self.tgt_stable_top_rot_pos,
                self.tgt_stable_top_rot_quat,
            )
            if (
                self._stage
                and self._asset_roots.get("TableTop")
                and self._ghost_paths_by_name.get("TableTop")
            ):
                update_ghost_preview_pose(
                    self._stage,
                    self._asset_roots["TableTop"],
                    self._ghost_paths_by_name["TableTop"],
                    self.tgt_stable_top_rot_pos,
                    self.tgt_stable_top_rot_quat,
                )

        return result
        # return True

    def _check_table_top_rotation(self) -> bool:
        tgt = self._target_poses.get("TableTop")
        live = self.get_live_part_pose("TableTop")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        result = pos_err <= 0.01 and ang_err <= 3.5
        if result:
            self._target_poses["FrontRightLeg"] = (
                self.tgt_front_right_leg_rot_pos,
                self.tgt_front_right_leg_rot_quat,
            )
            self._target_poses["FrontLeftLeg"] = (
                self.tgt_front_left_leg_rot_pos,
                self.tgt_front_left_leg_rot_quat,
            )

            if (
                self._stage
                and self._asset_roots.get("FrontRightLeg")
                and self._ghost_paths_by_name.get("FrontRightLeg")
            ):
                update_ghost_preview_pose(
                    self._stage,
                    self._asset_roots["FrontRightLeg"],
                    self._ghost_paths_by_name["FrontRightLeg"],
                    self.tgt_front_right_leg_rot_pos,
                    self.tgt_front_right_leg_rot_quat,
                )

            if (
                self._stage
                and self._asset_roots.get("FrontLeftLeg")
                and self._ghost_paths_by_name.get("FrontLeftLeg")
            ):
                update_ghost_preview_pose(
                    self._stage,
                    self._asset_roots["FrontLeftLeg"],
                    self._ghost_paths_by_name["FrontLeftLeg"],
                    self.tgt_front_left_leg_rot_pos,
                    self.tgt_front_left_leg_rot_quat,
                )

        return result

    def _check_back_right_leg_insert(self) -> bool:
        tgt = self._target_poses.get("BackRightLeg")
        live = self.get_live_part_pose("BackRightLeg")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 and ang_err <= 3.5
        # return True

    def _check_back_left_leg_insert(self) -> bool:
        tgt = self._target_poses.get("BackLeftLeg")
        live = self.get_live_part_pose("BackLeftLeg")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 and ang_err <= 3.5
        # return True

    def is_final_assembly_valid(self) -> bool:
        return (
            self._check_table_top_rotation()
            and self._check_front_right_leg_insert()
            and self._check_front_left_leg_insert()
            and self._check_back_right_leg_insert()
            and self._check_back_left_leg_insert()
        )

    def final_unmet_constraints(self) -> List[Tuple[str, str]]:
        issues: List[Tuple[str, str]] = []

        if not self._check_table_top_rotation():
            issues.append(("TableTop", "Table Top is not aligned in the corner (Step 2)"))
        if not self._check_front_right_leg_insert():
            issues.append(("FrontRightLeg", "Front Right Leg is not aligned (Step 3)"))
        if not self._check_front_left_leg_insert():
            issues.append(("FrontLeftLeg", "Front Left Leg is not aligned (Step 4)"))
        if not self._check_back_right_leg_insert():
            issues.append(("BackRightLeg", "Back Right Leg is not aligned (Step 6)"))
        if not self._check_back_left_leg_insert():
            issues.append(("BackLeftLeg", "Back Left Leg is not aligned (Step 7)"))

        return issues
    
    def on_step_completed(self, env, step_index: int) -> None:

        # Step 4 complete
        if step_index == 3:
            self._target_poses["TableTop"] = (self.tgt_stable_top_rot_pos, self.tgt_stable_top_rot_quat)

            if (
                self._stage
                and self._asset_roots.get("TableTop")
                and self._ghost_paths_by_name.get("TableTop")
            ):
                update_ghost_preview_pose(
                    self._stage,
                    self._asset_roots["TableTop"],
                    self._ghost_paths_by_name["TableTop"],
                    self.tgt_stable_top_rot_pos,
                    self.tgt_stable_top_rot_quat,
                )
            return

        # Step 5 complete 
        if step_index == 4:
            self._target_poses["FrontRightLeg"] = (self.tgt_front_right_leg_rot_pos, self.tgt_front_right_leg_rot_quat)
            self._target_poses["FrontLeftLeg"]  = (self.tgt_front_left_leg_rot_pos,  self.tgt_front_left_leg_rot_quat)

            if (
                self._stage
                and self._asset_roots.get("FrontRightLeg")
                and self._ghost_paths_by_name.get("FrontRightLeg")
            ):
                update_ghost_preview_pose(
                    self._stage,
                    self._asset_roots["FrontRightLeg"],
                    self._ghost_paths_by_name["FrontRightLeg"],
                    self.tgt_front_right_leg_rot_pos,
                    self.tgt_front_right_leg_rot_quat,
                )

            if (
                self._stage
                and self._asset_roots.get("FrontLeftLeg")
                and self._ghost_paths_by_name.get("FrontLeftLeg")
            ):
                update_ghost_preview_pose(
                    self._stage,
                    self._asset_roots["FrontLeftLeg"],
                    self._ghost_paths_by_name["FrontLeftLeg"],
                    self.tgt_front_left_leg_rot_pos,
                    self.tgt_front_left_leg_rot_quat,
                )

            self.snap_parts_to_targets(env, ["FrontRightLeg", "FrontLeftLeg"])
            return
