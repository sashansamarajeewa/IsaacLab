from .base import BaseGuide, VisualSequenceHighlighter
from pxr import UsdGeom, Usd, UsdPhysics, Gf
import math
from typing import Optional

# PhysX runtime API (continuous rigid body transform during simulation)
# Docs: omni.physx.bindings._physx.PhysX.get_rigidbody_transformation(stage, prim_path)
from omni.physx.bindings._physx import PhysX as _PhysX


# ----------------------- small helpers -----------------------

def _ang_deg(q1: Gf.Quatd, q2: Gf.Quatd) -> float:
    """Smallest angle (deg) between two quaternions."""
    dq = q1 * q2.GetInverse()
    a = 2.0 * math.degrees(math.acos(max(-1.0, min(1.0, abs(dq.GetReal())))))
    return min(a, 360.0 - a)

def _first_descendant_with_rigid_body(stage: Usd.Stage, root_prim: Usd.Prim) -> Optional[Usd.Prim]:
    """Return the first descendant (including root) that has UsdPhysics.RigidBodyAPI."""
    if not root_prim or not root_prim.IsValid():
        return None
    if UsdPhysics.RigidBodyAPI(root_prim):
        return root_prim
    for p in Usd.PrimRange(root_prim):
        if UsdPhysics.RigidBodyAPI(p):
            return p
    return None

def _resolve_env_scoped_path(stage: Usd.Stage, env_root_path: str, leaf_name: str) -> Optional[str]:
    """Resolve prim path for leaf_name under env root. Bounded search (no full-stage scan)."""
    exact = stage.GetPrimAtPath(f"{env_root_path}/{leaf_name}")
    if exact and exact.IsValid():
        return str(exact.GetPath())
    env_root = stage.GetPrimAtPath(env_root_path)
    if not env_root or not env_root.IsValid():
        return None
    for p in Usd.PrimRange(env_root):
        if p.GetName() == leaf_name:
            return str(p.GetPath())
    return None

def _physx_get_pose(stage: Usd.Stage, prim_path: str) -> Optional[tuple[Gf.Vec3d, Gf.Quatd]]:
    """
    Live PhysX pose for a rigid body prim using PhysX.get_rigidbody_transformation.

    Expected return layout from PhysX:
        (x, y, z, qx, qy, qz, qw)  # per NVIDIA docs & forum examples
    If your build returns (x, y, z, qw, qx, qy, qz), swap accordingly.
    """
    if not prim_path:
        return None
    try:
        x, y, z, qx, qy, qz, qw = _PhysX.get_rigidbody_transformation(stage, prim_path)
        return Gf.Vec3d(float(x), float(y), float(z)), Gf.Quatd(float(qw), float(qx), float(qy), float(qz))
    except Exception:
        return None


# ======================= Drawer Guide =======================

class DrawerGuide(BaseGuide):
    """
    Multi-step flow (keeps DrawerBox highlighted across first 2 steps):
      0) Pick up DrawerBox
      1) Brace DrawerBox on ObstacleLeft
      2) Insert DrawerBottom
      3) Insert DrawerTop
    """
    SEQUENCE = ["DrawerBox", "DrawerBox", "DrawerBottom", "DrawerTop"]

    def __init__(self):
        super().__init__()
        self._checks = [
            self._check_pickup_box,
            self._check_braced_box,
            self._check_bottom_insert,
            self._check_top_insert,
        ]
        # Cached prim paths (resolved on reset). Moving parts point to rigid body prims.
        self._paths: dict[str, Optional[str]] = {}

    # ------------------- lifecycle / reset -------------------

    def on_reset(self, env):
        stage: Usd.Stage = env.scene.stage
        env_ns: str = env.scene.env_ns  # e.g., "/World/envs/env_0"
        self._paths.clear()

        # Table can be named "PackingTable" or "Table" — normalize to key "Table"
        table_path = _resolve_env_scoped_path(stage, env_ns, "PackingTable") \
            or _resolve_env_scoped_path(stage, env_ns, "Table")
        self._paths["Table"] = table_path

        # Obstacles (static)
        for name in ("ObstacleLeft", "ObstacleFront"):
            self._paths[name] = _resolve_env_scoped_path(stage, env_ns, name)

        # Moving parts → resolve to rigid body prim (child) if present; else keep root
        for name in ("DrawerBox", "DrawerBottom", "DrawerTop"):
            root_path = _resolve_env_scoped_path(stage, env_ns, name)
            if not root_path:
                self._paths[name] = None
                continue
            rb_prim = _first_descendant_with_rigid_body(stage, stage.GetPrimAtPath(root_path))
            self._paths[name] = str(rb_prim.GetPath()) if rb_prim and rb_prim.IsValid() else root_path

        print("[Guide] Resolved prim paths:", self._paths)

    # ---------------------- HUD content ----------------------

    def step_label(self, highlighter: VisualSequenceHighlighter) -> str:
        idx, total = highlighter.step_index, (highlighter.total_steps or 1)
        return (
            f"Step 1/{total}: Pick up DrawerBox" if idx == 0 else
            f"Step 2/{total}: Brace DrawerBox against left obstacle" if idx == 1 else
            f"Step 3/{total}: Insert DrawerBottom into DrawerBox" if idx == 2 else
            f"Step 4/{total}: Insert DrawerTop into DrawerBox" if idx == 3 else
            "Assembly complete!"
        )

    # ------------------ per-frame evaluation -----------------

    def maybe_auto_advance(self, env, highlighter: VisualSequenceHighlighter):
        """
        Call once per sim tick *after* env.step() or sim.render().
        For moving parts, reads live PhysX poses; for static refs, reads USD xform.
        """
        idx = highlighter.step_index
        if idx >= len(self._checks):
            return

        stage: Usd.Stage = env.scene.stage
        # Fresh cache for static USD reads (table/obstacles)
        cache = UsdGeom.XformCache()

        if self._checks[idx](env, stage, cache):
            highlighter.advance()

    # ---------------------- pose getters ---------------------

    def _get_table_pos(self, stage: Usd.Stage, cache: UsdGeom.XformCache) -> Optional[Gf.Vec3d]:
        p = self._paths.get("Table")
        if not p:
            return None
        prim = stage.GetPrimAtPath(p)
        if not prim or not prim.IsValid():
            return None
        xf = cache.GetLocalToWorldTransform(prim)
        return xf.ExtractTranslation()

    def _get_obstacle_pose(self, name: str, stage: Usd.Stage, cache: UsdGeom.XformCache) -> Optional[tuple[Gf.Vec3d, Gf.Quatd]]:
        p = self._paths.get(name)
        if not p:
            return None
        prim = stage.GetPrimAtPath(p)
        if not prim or not prim.IsValid():
            return None
        xf = cache.GetLocalToWorldTransform(prim)
        return xf.ExtractTranslation(), xf.ExtractRotation().GetQuat()

    def _get_live_part_pose(self, name: str, stage: Usd.Stage) -> Optional[tuple[Gf.Vec3d, Gf.Quatd]]:
        """
        Live pose for moving parts: DrawerBox / DrawerBottom / DrawerTop.
        Uses PhysX.get_rigidbody_transformation on the resolved rigid body prim.
        """
        p = self._paths.get(name)
        if not p:
            return None
        return _physx_get_pose(stage, p)

    # ---------------------- step checks ----------------------

    def _check_pickup_box(self, env, stage, cache) -> bool:
        table_pos = self._get_table_pos(stage, cache)
        box_pose  = self._get_live_part_pose("DrawerBox", stage)
        if not (table_pos and box_pose):
            return False
        box_pos, _ = box_pose
        # ≥ 6 cm above table
        return (box_pos[2] - table_pos[2]) >= 1.1

    def _check_braced_box(self, env, stage, cache) -> bool:
        obs_pose = self._get_obstacle_pose("ObstacleLeft", stage, cache)
        box_pose = self._get_live_part_pose("DrawerBox", stage)
        if not (obs_pose and box_pose):
            return False
        obs_pos, obs_quat = obs_pose
        box_pos, box_quat = box_pose
        d = (obs_pos - box_pos).GetLength()
        ang = _ang_deg(obs_quat, box_quat)
        return (d <= 0.03) and (ang <= 15.0)

    def _check_bottom_insert(self, env, stage, cache) -> bool:
        box_pose = self._get_live_part_pose("DrawerBox", stage)
        bot_pose = self._get_live_part_pose("DrawerBottom", stage)
        if not (box_pose and bot_pose):
            return False
        box_pos, box_quat = box_pose
        bot_pos, bot_quat = bot_pose
        d = (box_pos - bot_pos).GetLength()
        ang = _ang_deg(box_quat, bot_quat)
        dz = bot_pos[2] - box_pos[2]
        # Near + roughly aligned + slightly below box reference (i.e., "inside")
        return (d <= 0.02) and (ang <= 12.0) and (dz <= -0.005)

    def _check_top_insert(self, env, stage, cache) -> bool:
        box_pose = self._get_live_part_pose("DrawerBox", stage)
        top_pose = self._get_live_part_pose("DrawerTop", stage)
        if not (box_pose and top_pose):
            return False
        box_pos, box_quat = box_pose
        top_pos, top_quat = top_pose
        d = (box_pos - top_pos).GetLength()
        ang = _ang_deg(box_quat, top_quat)
        dz = top_pos[2] - box_pos[2]
        # Near + roughly aligned + slightly above box reference (i.e., "on top")
        return (d <= 0.02) and (ang <= 12.0) and (dz >= 0.005)
