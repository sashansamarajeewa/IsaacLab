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

# ------------------- Lamp Guide -------------------


class LampGuide(BaseGuide):

    SEQUENCE = ["LampBase", "LampBase", "LampBulb", "LampHood"]
    MOVING_PARTS = ("LampBase", "LampBulb", "LampHood")
    STATIC_PARTS = ("ObstacleLeft", "ObstacleFront", "ObstacleRight")

    tol_z_dbox_t = 1.082  # distance between lamp base and table origin along Z

    tgt_base_pos = Gf.Vec3d(-0.26464322209358215, 0.5353000164031982, 1.0284510850906372)
    tgt_base_quat = Gf.Quatd(-0.5746305584907532, Gf.Vec3d(-0.5746316909790039, 0.4120660722255707, -0.4120675325393677))
    tgt_bulb_pos = Gf.Vec3d(-0.26490530371665955, 0.535226047039032, 1.164040446281433)
    tgt_bulb_quat = Gf.Quatd(0.7070314288139343, Gf.Vec3d(0.7071546912193298, -0.005556972697377205, -0.002830658107995987))
    tgt_hood_pos = Gf.Vec3d(-0.26467111706733704, 0.5362342000007629, 1.2153171300888062)
    tgt_hood_quat = Gf.Quatd(0.10603450238704681, Gf.Vec3d(-0.10304032266139984, 0.6999038457870483, -0.6987661719322205))

    def __init__(self):
        super().__init__()
        self._checks = [
            self._check_pickup_base,
            self._check_braced_base,
            self._check_bulb_insert,
            self._check_hood_insert,
        ]
        # Resolved prim paths. Moving parts - rigid body prim if available
        self._paths: dict[str, Optional[str]] = {}
        # Asset root paths for ghosts
        self._asset_roots: dict[str, Optional[str]] = {
            "LampBase": None,
            "LampBulb": None,
            "LampHood": None,
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
            "LampBase": None,
            "LampBulb": None,
            "LampHood": None,
        }

        # Ghost prim paths by logical name
        self._ghost_paths_by_name: dict[str, str] = {}

    # ------------------- reset -------------------

    def on_reset(self, env):
        super().on_reset(env)
        stage: Usd.Stage = env.scene.stage
        env_ns: str = env.scene.env_ns
        self._paths.clear()
        self._asset_roots = {"LampBase": None, "LampBulb": None, "LampHood": None}
        self._target_poses = {
            "LampBase": None,
            "LampBulb": None,
            "LampHood": None,
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

            # target LampBase braced in corner
            self._target_poses["LampBase"] = (self.tgt_base_pos, self.tgt_base_quat)

            # target LampBulb inserted to LampBase
            self._target_poses["LampBulb"] = (self.tgt_bulb_pos, self.tgt_bulb_quat)

            # target LampHood inserted to LampBase
            self._target_poses["LampHood"] = (self.tgt_hood_pos, self.tgt_hood_quat)

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
            f"Step 1/{total}: Pick up Lamp Base",
            f"Step 2/{total}: Brace Lamp Base against the front and left corner obstacles",
            f"Step 3/{total}: Insert Lamp Bulb into Lamp Base and screw clockwise until tight",
            f"Step 4/{total}: Place Lamp Hood on top of Lamp Base",
        ]
        base_steps.append("Assembly complete!")
        return base_steps

    # ---------------------- checks ----------------------

    def _check_pickup_base(self) -> bool:
        if self._static_table_pos is None:
            return False
        base_pose = self.get_live_part_pose("LampBase")
        if not base_pose:
            return False
        base_pos, _ = base_pose
        return (base_pos[2] - self._static_table_pos[2]) >= self.tol_z_dbox_t

    def _check_braced_base(self) -> bool:
        tgt = self._target_poses.get("LampBase")
        live = self.get_live_part_pose("LampBase")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 #and ang_err <= 3.0

    def _check_bulb_insert(self) -> bool:
        tgt = self._target_poses.get("LampBulb")
        live = self.get_live_part_pose("LampBulb")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 #and ang_err <= 3.0

    def _check_hood_insert(self) -> bool:
        tgt = self._target_poses.get("LampHood")
        live = self.get_live_part_pose("LampHood")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)

        return pos_err <= 0.01 #and ang_err <= 3.0

    def is_final_assembly_valid(self) -> bool:
        return (
            self._check_braced_base()
            and self._check_bulb_insert()
            and self._check_hood_insert()
        )

    def final_unmet_constraints(self) -> List[Tuple[str, str]]:
        issues: List[Tuple[str, str]] = []

        if not self._check_braced_base():
            issues.append(
                ("LampBase", "Lamp Base is not aligned in the corner (Step 2)")
            )
        if not self._check_bulb_insert():
            issues.append(("LampBulb", "Lamp Bulb is not aligned (Step 3)"))
        if not self._check_hood_insert():
            issues.append(("LampHood", "Lamp Hood is not aligned (Step 4)"))

        return issues
