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

# ------------------- Stool Guide -------------------


class StoolGuide(BaseGuide):

    SEQUENCE = [
        "Seat",
        "Seat",
        "FirstLeg",
        "SecondLeg",
        "ThirdLeg",
    ]
    MOVING_PARTS = (
        "Seat",
        "FirstLeg",
        "SecondLeg",
        "ThirdLeg",
    )
    STATIC_PARTS = ("ObstacleLeft", "ObstacleFront", "ObstacleRight")

    tol_z_dbox_t = 1.082  # distance between stool seat and table origin along Z

    tgt_seat_pos = Gf.Vec3d(
        0.15217958390712738, 0.47628986835479736, 1.0210511684417725
    )
    tgt_seat_quat = Gf.Quatd(0.7068566083908081, Gf.Vec3d(0.7068567872047424, 0.018804030492901802, 0.01880406215786934))
    tgt_first_leg_pos = Gf.Vec3d(0.15560699999332428, 0.4026220440864563, 1.1019830703735352)
    tgt_first_leg_quat = Gf.Quatd(-0.503440797328949, Gf.Vec3d(-0.5042394399642944, -0.496090292930603, -0.4961698055267334))
    tgt_second_leg_pos = Gf.Vec3d(0.08582456409931183, 0.508266031742096, 1.1020972728729248)
    tgt_second_leg_quat = Gf.Quatd(-0.6906436085700989, Gf.Vec3d(-0.6918016076087952, 0.14913704991340637, 0.14892998337745667))
    tgt_third_leg_pos = Gf.Vec3d(0.21428146958351135, 0.5158480405807495, 1.101969599723816)
    tgt_third_leg_quat = Gf.Quatd(0.17381539940834045, Gf.Vec3d(0.17399808764457703, -0.6853842735290527, -0.6853915452957153))

    def __init__(self):
        super().__init__()
        self._checks = [
            self._check_pickup_seat,
            self._check_seat_position,
            self._check_first_leg_insert,
            self._check_second_leg_insert,
            self._check_third_leg_insert,
        ]
        # Resolved prim paths. Moving parts - rigid body prim if available
        self._paths: dict[str, Optional[str]] = {}
        # Asset root paths for ghosts
        self._asset_roots: dict[str, Optional[str]] = {
            "Seat": None,
            "FirstLeg": None,
            "SecondLeg": None,
            "ThirdLeg": None,
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
            "FirstLeg": None,
            "SecondLeg": None,
            "ThirdLeg": None,
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
            "FirstLeg": None,
            "SecondLeg": None,
            "ThirdLeg": None,
        }
        self._target_poses = {
            "Seat": None,
            "FirstLeg": None,
            "SecondLeg": None,
            "ThirdLeg": None,
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
                self.tgt_seat_pos,
                self.tgt_seat_quat,
            )

            # target FirstLeg inserted to Seat
            self._target_poses["FirstLeg"] = (
                self.tgt_first_leg_pos,
                self.tgt_first_leg_quat,
            )

            # target SecondLeg inserted to Seat
            self._target_poses["SecondLeg"] = (
                self.tgt_second_leg_pos,
                self.tgt_second_leg_quat,
            )

            # target ThirdLeg inserted to Seat
            self._target_poses["ThirdLeg"] = (
                self.tgt_third_leg_pos,
                self.tgt_third_leg_quat,
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
            f"Step 2/{total}: Move Seat to the target position",
            f"Step 3/{total}: Insert First Leg and screw clockwise until tight",
            f"Step 4/{total}: Insert Second Leg and screw clockwise until tight",
            f"Step 5/{total}: Insert Third Leg and screw clockwise until tight",
        ]
        base_steps.append("Assembly complete!")
        return base_steps

    # ---------------------- checks ----------------------

    def _check_pickup_seat(self) -> bool:
        if self._static_table_pos is None:
            return False
        top_pose = self.get_live_part_pose("Seat")
        if not top_pose:
            return False
        top_pos, _ = top_pose
        return (top_pos[2] - self._static_table_pos[2]) >= self.tol_z_dbox_t
        # return True

    def _check_seat_position(self) -> bool:
        tgt = self._target_poses.get("Seat")
        live = self.get_live_part_pose("Seat")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 and ang_err <= 3.5
        #return True

    def _check_first_leg_insert(self) -> bool:
        tgt = self._target_poses.get("FirstLeg")
        live = self.get_live_part_pose("FirstLeg")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 and ang_err <= 3.5
        #return True

    def _check_second_leg_insert(self) -> bool:
        tgt = self._target_poses.get("SecondLeg")
        live = self.get_live_part_pose("SecondLeg")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 and ang_err <= 3.5
        #return True

    def _check_third_leg_insert(self) -> bool:
        tgt = self._target_poses.get("ThirdLeg")
        live = self.get_live_part_pose("ThirdLeg")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 and ang_err <= 3.5

    def is_final_assembly_valid(self) -> bool:
        return (
            self._check_seat_position()
            and self._check_first_leg_insert()
            and self._check_second_leg_insert()
            and self._check_third_leg_insert()
        )

    def final_unmet_constraints(self) -> List[Tuple[str, str]]:
        issues: List[Tuple[str, str]] = []

        if not self._check_seat_position():
            issues.append(("Seat", "Seat is not aligned in the target (Step 2)"))
        if not self._check_first_leg_insert():
            issues.append(("FirstLeg", "First Leg is not aligned (Step 3)"))
        if not self._check_second_leg_insert():
            issues.append(("SecondLeg", "Second Leg is not aligned (Step 4)"))
        if not self._check_third_leg_insert():
            issues.append(("ThirdLeg", "Third Leg is not aligned (Step 5)"))

        return issues
