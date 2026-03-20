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

# ------------------- Chair Guide -------------------


class ChairGuide(BaseGuide):

    SEQUENCE = [
        "Seat",
        "Seat",
        "FrontRightLeg",
        "FrontLeftLeg",
        "Seat",
        "Back",
        "RightNut",
        "LeftNut",
    ]
    MOVING_PARTS = (
        "Seat",
        "FrontRightLeg",
        "FrontLeftLeg",
        "Back",
        "RightNut",
        "LeftNut",
    )
    STATIC_PARTS = ("ObstacleLeft", "ObstacleFront", "ObstacleRight")
    
    SCENE_KEY_MAP = {
        "Seat": "chair_seat",
        "FrontRightLeg": "chair_leg2",
        "FrontLeftLeg": "chair_leg1",
        "Back": "chair_back",
        "RightNut": "chair_nut2",
        "LeftNut": "chair_nut1",
    }

    SNAP_PLAN = {
        1: ["Seat"],
        2: ["Seat", "FrontRightLeg"],
        3: ["Seat", "FrontRightLeg", "FrontLeftLeg"],
        4: ["Seat", "FrontRightLeg", "FrontLeftLeg"],
        5: ["Seat", "FrontRightLeg", "FrontLeftLeg", "Back"],
        6: ["Seat", "FrontRightLeg", "FrontLeftLeg", "Back", "RightNut"],
        7: ["Seat", "FrontRightLeg", "FrontLeftLeg", "Back", "RightNut", "LeftNut"],
    }

    tol_z_dbox_t = 1.08  # distance between seat and table origin along Z

    tgt_chair_seat_pos = Gf.Vec3d(
        0.24999697506427765, 0.4554999768733978, 1.025051236152649
    )
    tgt_chair_seat_quat = Gf.Quatd(
        -3.8872713048476726e-08,
        Gf.Vec3d(-3.64379957318306e-08, -0.7071069478988647, -0.7071066498756409),
    )
    tgt_front_right_leg_pos = Gf.Vec3d(
        0.3172641396522522, 0.4174909293651581, 1.1140486001968384
    )
    tgt_front_right_leg_quat = Gf.Quatd(
        -0.4880022406578064,
        Gf.Vec3d(-0.4828118085861206, 0.5117199420928955, 0.5116987228393555),
    )
    tgt_front_left_leg_pos = Gf.Vec3d(
        0.18242791295051575, 0.41762053966522217, 1.1140278577804565
    )
    tgt_front_left_leg_quat = Gf.Quatd(
        0.5012286305427551,
        Gf.Vec3d(0.5016393661499023, -0.4989506006240845, -0.4981728792190552),
    )
    tgt_chair_seat_pos_rot = Gf.Vec3d(
        0.06693419069051743, 0.4598131775856018, 1.0646365880966187
    )
    tgt_chair_seat_quat_rot = Gf.Quatd(
        -0.011706930585205555,
        Gf.Vec3d(0.00017946516163647175, 0.00010747313353931531, -0.9999314546585083),
    )
    tgt_front_right_leg_pos_rot = Gf.Vec3d(
        0.13236378133296967, 0.369273841381073, 1.0265432596206665
    )
    tgt_front_right_leg_quat_rot = Gf.Quatd(
        0.008277300745248795,
        Gf.Vec3d(-0.7160747051239014, 0.008312846533954144, 0.6979252099990845),
    )
    tgt_front_left_leg_pos_rot = Gf.Vec3d(
        -0.0024638744071125984, 0.3724214434623718, 1.0264832973480225
    )
    tgt_front_left_leg_quat_rot = Gf.Quatd(
        0.008160511963069439,
        Gf.Vec3d(-0.7239612936973572, 0.00832752138376236, 0.6897421479225159),
    )
    tgt_chair_back_pos = Gf.Vec3d(
        0.06832481920719147, 0.5259986519813538, 1.1630704402923584
    )
    tgt_chair_back_quat = Gf.Quatd(
        -0.015152636915445328,
        Gf.Vec3d(-0.9998193979263306, 0.0114845996722579, -3.826577085419558e-06),
    )
    tgt_right_nut_pos = Gf.Vec3d(
        0.13664162158966064, 0.4590485692024231, 1.2272306680679321
    )
    tgt_right_nut_quat = Gf.Quatd(
        -0.03958119824528694,
        Gf.Vec3d(0.04283810406923294, -0.7051147818565369, 0.7066904306411743),
    )
    tgt_left_nut_pos = Gf.Vec3d(
        -0.003298945492133498, 0.4623400866985321, 1.2271881103515625
    )
    tgt_left_nut_quat = Gf.Quatd(
        -0.03747441619634628,
        Gf.Vec3d(0.041586339473724365, -0.7052389979362488, 0.7067561745643616),
    )

    def __init__(self):
        super().__init__()
        self._checks = [
            self._check_pickup_seat,
            self._check_braced_seat,
            self._check_front_right_leg_insert,
            self._check_front_left_leg_insert,
            self._check_seat_rotation,
            self._check_back_insert,
            self._check_right_nut_insert,
            self._check_left_nut_insert,
        ]
        # Resolved prim paths. Moving parts - rigid body prim if available
        self._paths: dict[str, Optional[str]] = {}
        # Asset root paths for ghosts
        self._asset_roots: dict[str, Optional[str]] = {
            "Seat": None,
            "FrontRightLeg": None,
            "FrontLeftLeg": None,
            "Back": None,
            "RightNut": None,
            "LeftNut": None,
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
            "Seat": None,
            "FrontRightLeg": None,
            "FrontLeftLeg": None,
            "Back": None,
            "RightNut": None,
            "LeftNut": None,
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
            "Seat": None,
            "FrontRightLeg": None,
            "FrontLeftLeg": None,
            "Back": None,
            "RightNut": None,
            "LeftNut": None,
        }
        self._target_poses = {
            "Seat": None,
            "FrontRightLeg": None,
            "FrontLeftLeg": None,
            "Back": None,
            "RightNut": None,
            "LeftNut": None,
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

            # target Seat braced in corner
            self._target_poses["Seat"] = (
                self.tgt_chair_seat_pos,
                self.tgt_chair_seat_quat,
            )

            # target FrontRightLeg inserted to Seat
            self._target_poses["FrontRightLeg"] = (
                self.tgt_front_right_leg_pos,
                self.tgt_front_right_leg_quat,
            )

            # target FrontLeftLeg inserted to Seat
            self._target_poses["FrontLeftLeg"] = (
                self.tgt_front_left_leg_pos,
                self.tgt_front_left_leg_quat,
            )

            # target Back inserted to Seat
            self._target_poses["Back"] = (
                self.tgt_chair_back_pos,
                self.tgt_chair_back_quat,
            )

            # target RightNut inserted to Seat
            self._target_poses["RightNut"] = (
                self.tgt_right_nut_pos,
                self.tgt_right_nut_quat,
            )

            # target LeftNut inserted to Seat
            self._target_poses["LeftNut"] = (
                self.tgt_left_nut_pos,
                self.tgt_left_nut_quat,
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
            f"Step 1/{total}: Pick up Seat",
            f"Step 2/{total}: Brace Seat against the front and right corner obstacles",
            f"Step 3/{total}: Insert Front Right Leg and screw clockwise until tight",
            f"Step 4/{total}: Insert Front Left Leg and screw clockwise until tight",
            f"Step 5/{total}: Rotate Seat by 90°",
            f"Step 6/{total}: Insert Back into Seat",
            f"Step 7/{total}: Insert Right Nut and screw clockwise until tight",
            f"Step 8/{total}: Insert Left Nut and screw clockwise until tight",
        ]
        base_steps.append("Assembly complete!")
        return base_steps

    # ---------------------- checks ----------------------

    def _check_pickup_seat(self) -> bool:
        if self._static_table_pos is None:
            return False
        seat_pose = self.get_live_part_pose("Seat")
        if not seat_pose:
            return False
        seat_pos, _ = seat_pose
        return (seat_pos[2] - self._static_table_pos[2]) >= self.tol_z_dbox_t
        # return True

    def _check_braced_seat(self) -> bool:
        tgt = self._target_poses.get("Seat")
        live = self.get_live_part_pose("Seat")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 and ang_err <= 3.5
        # return True

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
        # return True

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
            self._target_poses["Seat"] = (
                self.tgt_chair_seat_pos_rot,
                self.tgt_chair_seat_quat_rot,
            )
            self._target_poses["FrontRightLeg"] = (
                self.tgt_front_right_leg_pos_rot,
                self.tgt_front_right_leg_quat_rot,
            )
            self._target_poses["FrontLeftLeg"] = (
                self.tgt_front_left_leg_pos_rot,
                self.tgt_front_left_leg_quat_rot,
            )
            if (
                self._stage
                and self._asset_roots.get("Seat")
                and self._ghost_paths_by_name.get("Seat")
            ):
                update_ghost_preview_pose(
                    self._stage,
                    self._asset_roots["Seat"],
                    self._ghost_paths_by_name["Seat"],
                    self.tgt_chair_seat_pos_rot,
                    self.tgt_chair_seat_quat_rot,
                )

        return result
        # return True

    def _check_seat_rotation(self) -> bool:
        tgt = self._target_poses.get("Seat")
        live = self.get_live_part_pose("Seat")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 and ang_err <= 3.5

    def _check_back_insert(self) -> bool:
        tgt = self._target_poses.get("Back")
        live = self.get_live_part_pose("Back")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 and ang_err <= 3.5
        # return True

    def _check_right_nut_insert(self) -> bool:
        tgt = self._target_poses.get("RightNut")
        live = self.get_live_part_pose("RightNut")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 and ang_err <= 8
        # return True

    def _check_left_nut_insert(self) -> bool:
        tgt = self._target_poses.get("LeftNut")
        live = self.get_live_part_pose("LeftNut")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 and ang_err <= 8
        # return True

    def is_final_assembly_valid(self) -> bool:
        return (
            self._check_seat_rotation()
            and self._check_back_insert()
            and self._check_right_nut_insert()
            and self._check_left_nut_insert()
        )

    def final_unmet_constraints(self) -> List[Tuple[str, str]]:
        issues: List[Tuple[str, str]] = []

        if not self._check_seat_rotation():
            issues.append(("Seat", "Seat is not aligned (Step 5)"))
        if not self._check_back_insert():
            issues.append(("Back", "Back is not aligned (Step 6)"))
        if not self._check_right_nut_insert():
            issues.append(("RightNut", "Right Nut is not aligned (Step 7)"))
        if not self._check_left_nut_insert():
            issues.append(("LeftNut", "Left Nut is not aligned (Step 8)"))

        return issues
    
    def on_step_completed(self, env, step_index: int) -> None:

        # Step 5 complete
        if step_index == 4:
            self._target_poses["Seat"] = (self.tgt_chair_seat_pos_rot, self.tgt_chair_seat_quat_rot)
            self._target_poses["FrontRightLeg"] = (self.tgt_front_right_leg_pos_rot, self.tgt_front_right_leg_quat_rot)
            self._target_poses["FrontLeftLeg"] = (self.tgt_front_left_leg_pos_rot, self.tgt_front_left_leg_quat_rot)

            # Update ghost preview for seat
            if (
                self._stage
                and self._asset_roots.get("Seat")
                and self._ghost_paths_by_name.get("Seat")
            ):
                update_ghost_preview_pose(
                    self._stage,
                    self._asset_roots["Seat"],
                    self._ghost_paths_by_name["Seat"],
                    self.tgt_chair_seat_pos_rot,
                    self.tgt_chair_seat_quat_rot,
                )

            self.snap_parts_to_targets(env, ["Seat", "FrontRightLeg", "FrontLeftLeg"])
            return
