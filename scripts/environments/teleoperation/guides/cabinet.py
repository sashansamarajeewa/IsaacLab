from .base import (
    BaseGuide,
    ang_deg,
    bind_base_white_for_moving_parts,
    first_descendant_with_rigid_body,
    resolve_env_scoped_path,
    spawn_ghost_preview,
    update_ghost_preview_pose,
    MaterialRegistry,
)
from pxr import UsdGeom, Usd, Gf
from typing import List, Optional, Tuple

# ------------------- Cabinet Guide -------------------


class CabinetGuide(BaseGuide):

    SEQUENCE = ["CabinetDoorLeft", "CabinetDoorRight", "CabinetBody", "CabinetTop"]
    MOVING_PARTS = ("CabinetBody", "CabinetDoorLeft", "CabinetDoorRight", "CabinetTop")
    STATIC_PARTS = ("ObstacleLeft", "ObstacleFront", "ObstacleRight")
    
    SCENE_KEY_MAP = {
        "CabinetBody": "cabinet_body",
        "CabinetDoorLeft": "cabinet_door_left",
        "CabinetDoorRight": "cabinet_door_right",
        "CabinetTop": "cabinet_top",
    }

    SNAP_PLAN = {
        0: ["CabinetBody", "CabinetDoorLeft"],
        1: ["CabinetBody", "CabinetDoorLeft", "CabinetDoorRight"],
        2: ["CabinetBody", "CabinetDoorLeft", "CabinetDoorRight"],
        3: ["CabinetBody", "CabinetDoorLeft", "CabinetDoorRight", "CabinetTop"],
    }

    tgt_ldoor_pos = Gf.Vec3d(-0.020940018817782402, 0.4008996784687042, 1.1067736148834229)
    tgt_ldoor_quat = Gf.Quatd(
        -0.00958466250449419,
        Gf.Vec3d(0.0050072926096618176, -0.9998884797096252, 0.010301811620593071),
    )
    tgt_rdoor_pos = Gf.Vec3d(-0.12322989851236343, 0.4011825919151306, 1.1069523096084595)
    tgt_rdoor_quat = Gf.Quatd(
        0.01594029739499092,
        Gf.Vec3d(-0.000999385374598205, -0.9997969269752502, 0.012297765351831913),
    )
    tgt_body_pos_initial = Gf.Vec3d(-0.07244517654180527, 0.4748646020889282, 1.0516512393951416)
    tgt_body_quat_initial = Gf.Quatd(
        0.7071068,
        Gf.Vec3d(0, 0.7071068, 0),
    )
    tgt_body_pos = Gf.Vec3d(-0.07455084472894669, 0.43206366896629333, 1.1391513347625732)
    tgt_body_quat = Gf.Quatd(
        0.49866804480552673,
        Gf.Vec3d(-0.49866822361946106, 0.5013284683227539, -0.5013283491134644),
    )
    tgt_ldoor_pos_rot = Gf.Vec3d(-0.022970162332057953, 0.48839521408081055, 1.212223768234253)
    tgt_ldoor_quat_rot = Gf.Quatd(
        -0.0028556538745760918,
        Gf.Vec3d(0.0017095773946493864, -0.7052011489868164, 0.708999514579773),
    )
    tgt_rdoor_pos_rot = Gf.Vec3d(-0.12662413716316223, 0.4957791268825531, 1.212080955505371)
    tgt_rdoor_quat_rot = Gf.Quatd(
        -0.04190424457192421,
        Gf.Vec3d(0.0444624125957489, -0.6982754468917847, 0.7132171392440796),
    )
    tgt_top_pos = Gf.Vec3d(-0.07441174983978271, 0.43180203437805176, 1.294395923614502)
    tgt_top_quat = Gf.Quatd(
        0.4777563214302063,
        Gf.Vec3d(-0.7237769961357117, 0.4260793924331665, 0.2575893998146057),
    )

    def __init__(self):
        super().__init__()
        self._checks = [
            self._check_insert_left_door,
            self._check_insert_right_door,
            self._check_body_rotation,
            self._check_top_insert,
        ]
        # Resolved prim paths. Moving parts - rigid body prim if available
        self._paths: dict[str, Optional[str]] = {}
        # Asset root paths for ghosts
        self._asset_roots: dict[str, Optional[str]] = {
            "CabinetBody": None,
            "CabinetDoorLeft": None,
            "CabinetDoorRight": None,
            "CabinetTop": None,
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
            "CabinetBody": None,
            "CabinetDoorLeft": None,
            "CabinetDoorRight": None,
            "CabinetTop": None,
        }

        # Ghost prim paths by logical name
        self._ghost_paths_by_name: dict[str, str] = {}

    # ------------------- reset -------------------

    def on_reset(self, env):
        super().on_reset(env)
        stage: Usd.Stage = env.scene.stage
        env_ns: str = env.scene.env_ns
        self._paths.clear()
        self._asset_roots = {"CabinetBody": None, "CabinetDoorLeft": None, "CabinetDoorRight": None, "CabinetTop": None}
        self._target_poses = {
            "CabinetBody": None,
            "CabinetDoorLeft": None,
            "CabinetDoorRight": None,
            "CabinetTop": None,
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

            # target CabinetBody braced in corner
            self._target_poses["CabinetBody"] = (self.tgt_body_pos_initial, self.tgt_body_quat_initial)

            # target CabinetDoorLeft inserted to DrawerBox
            self._target_poses["CabinetDoorLeft"] = (self.tgt_ldoor_pos, self.tgt_ldoor_quat)

            # target CabinetDoorRight inserted to DrawerBox
            self._target_poses["CabinetDoorRight"] = (self.tgt_rdoor_pos, self.tgt_rdoor_quat)

            # target CabinetTop inserted to DrawerBox
            self._target_poses["CabinetTop"] = (self.tgt_top_pos, self.tgt_top_quat)

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
            f"Step 1/{total}: Pick and insert Cabinet Door Left into Cabinet Body",
            f"Step 2/{total}: Pick and insert Cabinet Door Right into Cabinet Body",
            f"Step 3/{total}: Rotate Cabinet Body by 90° to face threads up",
            f"Step 4/{total}: Insert Cabinet Top and screw clockwise until tight",
        ]
        base_steps.append("Assembly complete!")
        return base_steps

    # ---------------------- checks ----------------------

    def _check_insert_left_door(self) -> bool:
        tgt = self._target_poses.get("CabinetDoorLeft")
        live = self.get_live_part_pose("CabinetDoorLeft")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 and ang_err <= 3.0

    def _check_insert_right_door(self) -> bool:
        tgt = self._target_poses.get("CabinetDoorRight")
        live = self.get_live_part_pose("CabinetDoorRight")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        result = pos_err <= 0.01 and ang_err <= 3.0
        if result:
            self._target_poses["CabinetBody"] = (
                self.tgt_body_pos,
                self.tgt_body_quat,
            )
            if (
                self._stage
                and self._asset_roots.get("CabinetBody")
                and self._ghost_paths_by_name.get("CabinetBody")
            ):
                update_ghost_preview_pose(
                    self._stage,
                    self._asset_roots["CabinetBody"],
                    self._ghost_paths_by_name["CabinetBody"],
                    self.tgt_body_pos,
                    self.tgt_body_quat,
                )
            self._target_poses["CabinetDoorLeft"] = (
                self.tgt_ldoor_pos_rot,
                self.tgt_ldoor_quat_rot,
            )
            self._target_poses["CabinetDoorRight"] = (
                self.tgt_rdoor_pos_rot,
                self.tgt_rdoor_quat_rot,
            )
                
        return result

    def _check_body_rotation(self) -> bool:
        tgt = self._target_poses.get("CabinetBody")
        live = self.get_live_part_pose("CabinetBody")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 and ang_err <= 3.0

    def _check_top_insert(self) -> bool:
        tgt = self._target_poses.get("CabinetTop")
        live = self.get_live_part_pose("CabinetTop")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 and ang_err <= 10.0

    def is_final_assembly_valid(self) -> bool:
        return (
            self._check_body_rotation()
            and self._check_top_insert()
        )

    def final_unmet_constraints(self) -> List[Tuple[str, str]]:
        issues: List[Tuple[str, str]] = []

        if not self._check_body_rotation():
            issues.append(("CabinetBody", "Cabinet Body is not aligned (Step 3)"))
        if not self._check_top_insert():
            issues.append(("CabinetTop", "Cabinet Top is not aligned (Step 4)"))

        return issues

    def on_step_completed(self, env, step_index: int) -> None:

        # Step 2 complete
        if step_index == 1:
            self._target_poses["CabinetBody"] = (self.tgt_body_pos, self.tgt_body_quat)
            self._target_poses["CabinetDoorLeft"] = (self.tgt_ldoor_pos_rot, self.tgt_ldoor_quat_rot)
            self._target_poses["CabinetDoorRight"] = (self.tgt_rdoor_pos_rot, self.tgt_rdoor_quat_rot)

            # Update ghost preview for CabinetBody
            if (
                self._stage
                and self._asset_roots.get("CabinetBody")
                and self._ghost_paths_by_name.get("CabinetBody")
            ):
                update_ghost_preview_pose(
                    self._stage,
                    self._asset_roots["CabinetBody"],
                    self._ghost_paths_by_name["CabinetBody"],
                    self.tgt_body_pos,
                    self.tgt_body_quat,
                )

            #self.snap_parts_to_targets(env, ["CabinetBody", "CabinetDoorLeft", "CabinetDoorRight"])
            return