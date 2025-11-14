from .base import BaseGuide, VisualSequenceHighlighter
from pxr import UsdGeom, Usd, UsdPhysics, Gf
from typing import Optional, Tuple
import math
from omni.physx import get_physx_interface

# ----------------------- helpers -----------------------

def _ang_deg(q1: Gf.Quatd, q2: Gf.Quatd) -> float:
    dq = q1 * q2.GetInverse()
    a = 2.0 * math.degrees(math.acos(max(-1.0, min(1.0, abs(dq.GetReal())))))
    return min(a, 360.0 - a)

def _first_descendant_with_rigid_body(stage: Usd.Stage, root_prim: Usd.Prim) -> Optional[Usd.Prim]:
    if not root_prim or not root_prim.IsValid():
        return None
    if UsdPhysics.RigidBodyAPI(root_prim):
        return root_prim
    for p in Usd.PrimRange(root_prim):
        if UsdPhysics.RigidBodyAPI(p):
            return p
    return None

def _resolve_env_scoped_path(stage: Usd.Stage, env_root_path: str, leaf_name: str) -> Optional[str]:
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

def _physx_get_pose(prim_path: str) -> Optional[Tuple[Gf.Vec3d, Gf.Quatd]]:
    """
    Live PhysX pose for a rigid body prim using
    get_physx_interface().get_rigidbody_transformation(prim_path).

    Returns (Gf.Vec3d position, Gf.Quatd rotation) or None.
    """
    if not prim_path:
        return None
    try:
        result = get_physx_interface().get_rigidbody_transformation(prim_path)
        ret = result["ret_val"]
        pos = result["position"]
        rot = result["rotation"]
        if not ret or pos is None or rot is None:
            return None
        px, py, pz = float(pos.x), float(pos.y), float(pos.z)
        qx, qy, qz, qw = float(rot.x), float(rot.y), float(rot.z), float(rot.w)
        return Gf.Vec3d(px, py, pz), Gf.Quatd(qw, qx, qy, qz)
    except Exception:
        return None

# ======================= Drawer Guide =======================

class DrawerGuide(BaseGuide):
    """
    Sequence:
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
        # Resolved prim paths (set in on_reset). Moving parts -> rigid body prim if available.
        self._paths: dict[str, Optional[str]] = {}
        # Cached static world poses for this episode
        self._static_table_pos: Optional[Gf.Vec3d] = None
        self._static_obstacles: dict[str, Optional[Tuple[Gf.Vec3d, Gf.Quatd]]] = {
            "ObstacleLeft": None,
            "ObstacleFront": None,
        }

    # ------------------- lifecycle / reset -------------------

    def on_reset(self, env):
        print("Called reset")
        stage: Usd.Stage = env.scene.stage
        env_ns: str = env.scene.env_ns
        self._paths.clear()
        self._static_table_pos = None
        self._static_obstacles = {"ObstacleLeft": None, "ObstacleFront": None}

        # Table (static) — accept PackingTable or Table
        table_path = _resolve_env_scoped_path(stage, env_ns, "PackingTable")
        self._paths["Table"] = table_path

        # Obstacles (static)
        for name in ("ObstacleLeft", "ObstacleFront"):
            self._paths[name] = _resolve_env_scoped_path(stage, env_ns, name)

        # Moving parts → rigid body prim if present, else root
        for name in ("DrawerBox", "DrawerBottom", "DrawerTop"):
            root_path = _resolve_env_scoped_path(stage, env_ns, name)
            if not root_path:
                self._paths[name] = None
                continue
            rb_prim = _first_descendant_with_rigid_body(stage, stage.GetPrimAtPath(root_path))
            self._paths[name] = str(rb_prim.GetPath()) if rb_prim and rb_prim.IsValid() else root_path

        # Cache static world poses once
        cache = UsdGeom.XformCache()
        if self._paths.get("Table"):
            prim = stage.GetPrimAtPath(self._paths["Table"])
            if prim and prim.IsValid():
                self._static_table_pos = cache.GetLocalToWorldTransform(prim).ExtractTranslation()

        for name in ("ObstacleLeft", "ObstacleFront"):
            p = self._paths.get(name)
            if not p:
                continue
            prim = stage.GetPrimAtPath(p)
            if prim and prim.IsValid():
                xf = cache.GetLocalToWorldTransform(prim)
                self._static_obstacles[name] = (xf.ExtractTranslation(), xf.ExtractRotation().GetQuat())

        print("[Guide] Resolved prim paths:", self._paths)

    # ---------------------- HUD content ----------------------

    def step_label(self, highlighter: VisualSequenceHighlighter) -> str:
        idx, total = highlighter.step_index, (highlighter.total_steps or 1)
        if idx == 0:
            return f"Step 1/{total}: Pick up DrawerBox (lift above table)."
        elif idx == 1:
            return f"Step 2/{total}: Brace DrawerBox against the left corner obstacle."
        elif idx == 2:
            return f"Step 3/{total}: Insert DrawerBottom into DrawerBox."
        elif idx == 3:
            return f"Step 4/{total}: Insert DrawerTop to finish."
        return "Assembly complete!"

    # ------------------ per-frame evaluation -----------------

    def maybe_auto_advance(self, env, highlighter: VisualSequenceHighlighter):
        """
        Call once per sim tick *after* env.step() or sim.render().
        Uses PhysX for moving parts, cached USD for statics.
        """
        idx = highlighter.step_index
        if idx >= len(self._checks):
            return
        stage: Usd.Stage = env.scene.stage
        # cache is not used by moving parts, but keep the signature for checks
        cache = UsdGeom.XformCache()
        if self._checks[idx](env, stage, cache):
            highlighter.advance()

    # ---------------------- live pose getters ----------------------

    def _get_live_part_pose(self, name: str, stage: Usd.Stage) -> Optional[Tuple[Gf.Vec3d, Gf.Quatd]]:
        p = self._paths.get(name)
        if not p:
            return None
        return _physx_get_pose(p)

    # ---------------------- checks ----------------------

    def _check_pickup_box(self, env, stage, cache) -> bool:
        if self._static_table_pos is None:
            return False
        box_pose = self._get_live_part_pose("DrawerBox", stage)
        if not box_pose:
            return False
        box_pos, _ = box_pose
        # 1.081 meters above table
        return (box_pos[2] - self._static_table_pos[2]) >= 1.081

    def _check_braced_box(self, env, stage, cache) -> bool:
        obs_pose = self._static_obstacles.get("ObstacleLeft")
        box_pose = self._get_live_part_pose("DrawerBox", stage)
        if not (obs_pose and box_pose):
            return False
        obs_pos, obs_quat = obs_pose
        box_pos, box_quat = box_pose
        d = (obs_pos - box_pos).GetLength()
        print("d:")
        print(d)
        ang = _ang_deg(obs_quat, box_quat)
        print("ang:")
        print(ang)
        return (d <= 0.16) and (ang <= 180.0)

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
        return (d <= 0.02) and (ang <= 12.0) and (dz >= 0.005)
