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

# ------------------- Rtable Guide -------------------


class RtableGuide(BaseGuide):

    SEQUENCE = ["RoundLeg", "RoundLeg", "RoundTableTop", "RoundSupport"]
    MOVING_PARTS = ("RoundLeg", "RoundSupport", "RoundTableTop")
    STATIC_PARTS = ("ObstacleLeft", "ObstacleFront", "ObstacleRight")
    
    SCENE_KEY_MAP = {
        "RoundLeg": "round_leg",
        "RoundSupport": "round_support",
        "RoundTableTop": "round_table_top",
    }

    SNAP_PLAN = {
        1: ["RoundLeg", "RoundSupport"],
        2: ["RoundLeg", "RoundSupport", "RoundTableTop"],
        3: ["RoundLeg", "RoundSupport", "RoundTableTop"],
    }

    tol_z_dbox_t = 1.13  # distance between round leg and table origin along Z

    tgt_leg_pos = Gf.Vec3d(0.21071575582027435, 0.347117155790329, 1.1228574514389038)
    tgt_leg_quat = Gf.Quatd(
        0.7062255144119263,
        Gf.Vec3d(-0.7060852646827698, -0.03716788813471794, 0.036162737756967545),
    )
    tgt_support_pos = Gf.Vec3d(0.21078689396381378, 0.34730324149131775, 1.0165512561798096)
    tgt_support_quat = Gf.Quatd(
        4.871253622695804e-09,
        Gf.Vec3d(-0.007517402991652489, -0.9999717473983765, 3.434251993894577e-09),
    )
    tgt_top_pos = Gf.Vec3d(-0.15009844303131104, 0.42000797390937805, 0.9965510368347168)
    tgt_top_quat = Gf.Quatd(
        -0.7240346670150757,
        Gf.Vec3d(2.5640474632382393e-08, -4.72591636935249e-08, 0.6897637844085693),
    )
    tgt_leg_pos_rot = Gf.Vec3d(-0.14968684315681458, 0.4193904995918274, 1.091030478477478)
    tgt_leg_quat_rot = Gf.Quatd(
        -0.4911705255508423,
        Gf.Vec3d(-0.49259623885154724, 0.5027142763137817, 0.5132046937942505),
    )
    tgt_support_pos_rot = Gf.Vec3d(-0.14877615869045258, 0.41815385222435, 1.1972640752792358)
    tgt_support_quat_rot = Gf.Quatd(
        0.9990598559379578,
        Gf.Vec3d(0.006261101458221674, 0.004303178749978542, 0.04268259555101395),
    )

    def __init__(self):
        super().__init__()
        self._checks = [
            self._check_pickup_leg,
            self._check_insert_leg,
            self._check_braced_top,
            self._check_support_insert,
        ]
        # Resolved prim paths. Moving parts - rigid body prim if available
        self._paths: dict[str, Optional[str]] = {}
        # Asset root paths for ghosts
        self._asset_roots: dict[str, Optional[str]] = {
            "RoundLeg": None,
            "RoundSupport": None,
            "RoundTableTop": None,
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
            "RoundLeg": None,
            "RoundSupport": None,
            "RoundTableTop": None,
        }

        # Ghost prim paths by logical name
        self._ghost_paths_by_name: dict[str, str] = {}

    # ------------------- reset -------------------

    def on_reset(self, env):
        super().on_reset(env)
        stage: Usd.Stage = env.scene.stage
        env_ns: str = env.scene.env_ns
        self._paths.clear()
        self._asset_roots = {"RoundLeg": None, "RoundSupport": None, "RoundTableTop": None}
        self._target_poses = {
            "RoundLeg": None,
            "RoundSupport": None,
            "RoundTableTop": None,
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

            # target RoundLeg braced in corner
            self._target_poses["RoundLeg"] = (self.tgt_leg_pos, self.tgt_leg_quat)

            # target RoundSupport inserted to RoundLeg
            self._target_poses["RoundSupport"] = (self.tgt_support_pos, self.tgt_support_quat)

            # target RoundTableTop inserted to RoundSupport
            self._target_poses["RoundTableTop"] = (self.tgt_top_pos, self.tgt_top_quat)

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
            f"Step 1/{total}: SKIP-Pick up Round Leg",
            f"Step 2/{total}: Insert Round Leg into Round Support and screw clockwise to tighten",
            f"Step 3/{total}: Brace Round Table Top against the front and left corner obstacles",
            f"Step 4/{total}: SKIP-Insert Round Support into Round Table Top and screw clockwise until tight to finish",
        ]
        base_steps.append("Assembly complete!")
        return base_steps

    # ---------------------- checks ----------------------

    def _check_pickup_leg(self) -> bool:
        if self._static_table_pos is None:
            return False
        box_pose = self.get_live_part_pose("RoundLeg")
        if not box_pose:
            return False
        box_pos, _ = box_pose
        return (box_pos[2] - self._static_table_pos[2]) >= self.tol_z_dbox_t

    def _check_insert_leg(self) -> bool:
        tgt = self._target_poses.get("RoundLeg")
        live = self.get_live_part_pose("RoundLeg")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01# and ang_err <= 3.0

    def _check_braced_top(self) -> bool:
        tgt = self._target_poses.get("RoundTableTop")
        live = self.get_live_part_pose("RoundTableTop")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        result = pos_err <= 0.01 #and ang_err <= 3.5

        if result:
            self._target_poses["RoundLeg"] = (
                self.tgt_leg_pos_rot,
                self.tgt_leg_quat_rot,
            )
            self._target_poses["RoundSupport"] = (
                self.tgt_support_pos_rot,
                self.tgt_support_quat_rot,
            )

            if (
                self._stage
                and self._asset_roots.get("RoundLeg")
                and self._ghost_paths_by_name.get("RoundLeg")
            ):
                update_ghost_preview_pose(
                    self._stage,
                    self._asset_roots["RoundLeg"],
                    self._ghost_paths_by_name["RoundLeg"],
                    self.tgt_leg_pos_rot,
                    self.tgt_leg_quat_rot,
                )

            if (
                self._stage
                and self._asset_roots.get("RoundSupport")
                and self._ghost_paths_by_name.get("RoundSupport")
            ):
                update_ghost_preview_pose(
                    self._stage,
                    self._asset_roots["RoundSupport"],
                    self._ghost_paths_by_name["RoundSupport"],
                    self.tgt_support_pos_rot,
                    self.tgt_support_quat_rot,
                )

        return result

    def _check_support_insert(self) -> bool:
        tgt = self._target_poses.get("RoundSupport")
        live = self.get_live_part_pose("RoundSupport")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 #and ang_err <= 3.0

    def is_final_assembly_valid(self) -> bool:
        return (
            self._check_braced_top()
            and self._check_support_insert()
        )

    def final_unmet_constraints(self) -> List[Tuple[str, str]]:
        issues: List[Tuple[str, str]] = []
        if not self._check_braced_top():
            issues.append(("RoundTableTop", "Round Table Top is not aligned (Step 3)"))
        if not self._check_support_insert():
            issues.append(("RoundSupport", "Round Support is not aligned (Step 4)"))

        return issues

    def on_step_completed(self, env, step_index: int) -> None:

        # Step 3 complete
        if step_index == 2:
            # Update RoundLeg and RoundSupport target to "rotated" pose
            self._target_poses["RoundLeg"] = (self.tgt_leg_pos_rot, self.tgt_leg_quat_rot)
            self._target_poses["RoundSupport"] = (self.tgt_support_pos_rot, self.tgt_support_quat_rot)

            # Update RoundLeg ghost preview pose
            if (
                self._stage
                and self._asset_roots.get("RoundLeg")
                and self._ghost_paths_by_name.get("RoundLeg")
            ):
                update_ghost_preview_pose(
                    self._stage,
                    self._asset_roots["RoundLeg"],
                    self._ghost_paths_by_name["RoundLeg"],
                    self.tgt_leg_pos_rot,
                    self.tgt_leg_quat_rot,
                )

            # Update RoundSupport ghost preview pose
            if (
                self._stage
                and self._asset_roots.get("RoundSupport")
                and self._ghost_paths_by_name.get("RoundSupport")
            ):
                update_ghost_preview_pose(
                    self._stage,
                    self._asset_roots["RoundSupport"],
                    self._ghost_paths_by_name["RoundSupport"],
                    self.tgt_support_pos_rot,
                    self.tgt_support_quat_rot,
                )
            return