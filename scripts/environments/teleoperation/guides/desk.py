from .base import (
    BaseGuide,
    ang_deg,
    bind_base_white_for_moving_parts,
    first_descendant_with_rigid_body,
    resolve_env_scoped_path,
    spawn_ghost_preview,
    MaterialRegistry,
)
from pxr import UsdGeom, Usd, Gf
from typing import List, Optional, Tuple

# ------------------- Desk Guide -------------------


class DeskGuide(BaseGuide):

    SEQUENCE = [
        "DeskTop",
        "DeskTop",
        "FrontRightLeg",
        "FrontLeftLeg",
        "DeskTop"
        "BackRightLeg",
        "BackLeftLeg"
    ]
    MOVING_PARTS = (
        "DeskTop",
        "FrontRightLeg",
        "FrontLeftLeg",
        "BackRightLeg",
        "BackLeftLeg"
    )
    STATIC_PARTS = ("ObstacleLeft", "ObstacleFront", "ObstacleRight")

    tol_z_dbox_t = 1.082  # distance between desk top and table origin along Z

    tgt_desk_top_pos = Gf.Vec3d(0.17141205072402954, 0.4924437999725342, 1.0206135511398315)
    tgt_desk_top_quat = Gf.Quatd(
        2.1194635337451473e-05,
        Gf.Vec3d(2.127042898791842e-05, -0.7071068286895752, -0.7071068286895752),
    )
    tgt_front_right_leg_pos = Gf.Vec3d(0.3075884282588959, 0.4075841009616852, 1.1332392692565918)
    tgt_front_right_leg_quat = Gf.Quatd(
        0.7070425748825073,
        Gf.Vec3d(0.7070440649986267, 0.009473255835473537, 0.009473560377955437),
    )
    tgt_front_left_leg_pos = Gf.Vec3d(0.03558668866753578, 0.40764927864074707, 1.1332330703735352)
    tgt_front_left_leg_quat = Gf.Quatd(
        0.7070895433425903,
        Gf.Vec3d(0.7070915699005127, 0.0047986614517867565, 0.0047972965985536575),
    )
    tgt_desk_top_pos_rot = Gf.Vec3d(0.03540593758225441, 0.4074826240539551, 1.1332770586013794)
    tgt_desk_top_quat_rot = Gf.Quatd(
        0.7069599628448486,
        Gf.Vec3d(0.7070682048797607, -0.011597963981330395, -0.011298765428364277),
    )

    def __init__(self):
        super().__init__()
        self._checks = [
            self._check_pickup_desk_top,
            self._check_braced_desk_top,
            self._check_front_right_leg_insert,
            self._check_front_left_leg_insert,
            self._check_desk_top_rotation,
            self._check_back_right_leg_insert,
            self._check_back_left_leg_insert,
        ]
        # Resolved prim paths. Moving parts - rigid body prim if available
        self._paths: dict[str, Optional[str]] = {}
        # Asset root paths for ghosts
        self._asset_roots: dict[str, Optional[str]] = {
            "DeskTop": None,
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
            "DeskTop": None,
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
            "DeskTop": None,
            "FrontRightLeg": None,
            "FrontLeftLeg": None,
            "BackRightLeg": None,
            "BackLeftLeg": None,
        }
        self._target_poses = {
            "DeskTop": None,
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

            # target DeskTop braced in corner
            self._target_poses["DeskTop"] = (self.tgt_desk_top_pos, self.tgt_desk_top_quat)

            # target FrontRightLeg inserted to DeskTop
            self._target_poses["FrontRightLeg"] = (self.tgt_front_right_leg_pos, self.tgt_front_right_leg_quat)

            # target FrontLeftLeg inserted to DeskTop
            self._target_poses["FrontLeftLeg"] = (self.tgt_front_left_leg_pos, self.tgt_front_left_leg_quat)
            
            # target BackRightLeg inserted to DeskTop
            self._target_poses["BackRightLeg"] = (self.tgt_front_left_leg_pos, self.tgt_front_left_leg_quat)

            # target BackLeftLeg inserted to DeskTop
            self._target_poses["BackLeftLeg"] = (self.tgt_front_left_leg_pos, self.tgt_front_left_leg_quat)


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
            f"Step 1/{total}: Pick up Desk Top",
            f"Step 2/{total}: Brace Desk Top against the front and right corner obstacles",
            f"Step 3/{total}: Insert Front Right Leg into Desk Top and screw clockwise until tight",
            f"Step 4/{total}: Insert Front Left Leg into Desk Top and screw clockwise until tight",
            f"Step 5/{total}: Rotate Desk Top 180°",
            f"Step 6/{total}: Insert Back Right Leg into Desk Top and screw clockwise until tight",
            f"Step 7/{total}: Insert Back Left Leg into Desk Top and screw clockwise until tight",
        ]
        base_steps.append("Assembly complete!")
        return base_steps

    # ---------------------- checks ----------------------

    def _check_pickup_desk_top(self) -> bool:
        if self._static_table_pos is None:
            return False
        box_pose = self.get_live_part_pose("DeskTop")
        if not box_pose:
            return False
        box_pos, _ = box_pose
        # return (box_pos[2] - self._static_table_pos[2]) >= self.tol_z_dbox_t
        return True

    def _check_braced_desk_top(self) -> bool:
        tgt = self._target_poses.get("DeskTop")
        live = self.get_live_part_pose("DeskTop")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        # return pos_err <= 0.01 and ang_err <= 3.0
        return True

    def _check_front_right_leg_insert(self) -> bool:
        tgt = self._target_poses.get("FrontRightLeg")
        live = self.get_live_part_pose("FrontRightLeg")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        # return pos_err <= 0.01 and ang_err <= 3.0
        return True

    def _check_front_left_leg_insert(self) -> bool:
        tgt = self._target_poses.get("FrontLeftLeg")
        live = self.get_live_part_pose("FrontLeftLeg")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        result = (pos_err <= 0.01 and ang_err <= 3.0)
        if(result):
            self._target_poses["DeskTop"] = (self.tgt_desk_top_pos_rot, self.tgt_desk_top_quat_rot)

        return result
        #return True
    
    def _check_desk_top_rotation(self) -> bool:
        tgt = self._target_poses.get("DeskTop")
        live = self.get_live_part_pose("DeskTop")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 and ang_err <= 3.0
    
    def _check_back_right_leg_insert(self) -> bool:
        print(self.get_live_part_pose("BackRightLeg"))
        tgt = self._target_poses.get("BackRightLeg")
        live = self.get_live_part_pose("BackRightLeg")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 and ang_err <= 3.0
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

        return pos_err <= 0.01 and ang_err <= 3.0
        #return True

    def is_final_assembly_valid(self) -> bool:
        return (
            self._check_braced_desk_top()
            and self._check_front_right_leg_insert()
            and self._check_front_left_leg_insert()
            and self._check_back_right_leg_insert()
            and self._check_back_left_leg_insert()
        )

    def final_unmet_constraints(self) -> List[Tuple[str, str]]:
        issues: List[Tuple[str, str]] = []

        if not self._check_desk_top_rotation():
            issues.append(
                ("DeskTop", "Desk Top is not aligned in the corner (Step 2)")
            )
        if not self._check_front_right_leg_insert():
            issues.append(("FrontRightLeg", "Front Right Leg is not aligned (Step 3)"))
        if not self._check_front_left_leg_insert():
            issues.append(("FrontLeftLeg", "Front Left Leg is not aligned (Step 4)"))
        if not self._check_back_left_leg_insert():
            issues.append(("BackRightLeg", "Back Right Leg is not aligned (Step 6)"))
        if not self._check_back_left_leg_insert():
            issues.append(("BackLeftLeg", "Back Left Leg is not aligned (Step 7)"))

        return issues
