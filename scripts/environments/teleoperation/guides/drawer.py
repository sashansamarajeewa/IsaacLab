from .base import BaseGuide, VisualSequenceHighlighter
from pxr import UsdGeom, Usd, UsdPhysics, Gf
import math
from typing import Optional

# ----------------------- math helpers -----------------------

def _ang_deg(q1: Gf.Quatd, q2: Gf.Quatd) -> float:
    """Smallest angle (deg) between two quaternions."""
    dq = q1 * q2.GetInverse()
    a = 2.0 * math.degrees(math.acos(max(-1.0, min(1.0, abs(dq.GetReal())))))
    return min(a, 360.0 - a)

# ----------------------- USD helpers ------------------------

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
    """
    Resolve a prim path for leaf_name under env root.
    Tries exact path; otherwise searches only within the env root (no full-stage scan).
    """
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

# ======================= Drawer Guide =======================

class DrawerGuide(BaseGuide):
    """
    Multi-step flow:
      0) Pick up DrawerBox                 (highlight DrawerBox)
      1) Brace DrawerBox on ObstacleLeft   (still highlight DrawerBox)
      2) Insert DrawerBottom               (highlight DrawerBottom)
      3) Insert DrawerTop                  (highlight DrawerTop)
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
        # Cached prim paths (resolved on reset). Moving parts point to rigid body child prims.
        self._paths: dict[str, Optional[str]] = {}
        self._warned_no_updates = False

    # ------------------- lifecycle / reset -------------------

    def on_reset(self, env):
        stage: Usd.Stage = env.scene.stage
        env_ns: str = env.scene.env_ns  # e.g., "/World/envs/env_0"

        self._paths.clear()

        # Accept either "PackingTable" or "Table"
        table_name = None
        for candidate in ("PackingTable", "Table"):
            path = _resolve_env_scoped_path(stage, env_ns, candidate)
            if path:
                table_name = candidate
                self._paths["Table"] = path  # normalize to "Table" key
                break
        if table_name is None:
            self._paths["Table"] = None

        # Obstacles (static)
        for name in ("ObstacleLeft", "ObstacleFront"):
            self._paths[name] = _resolve_env_scoped_path(stage, env_ns, name)

        # Moving parts -> resolve to rigid body prims when possible
        for name in ("DrawerBox", "DrawerBottom", "DrawerTop"):
            root_path = _resolve_env_scoped_path(stage, env_ns, name)
            if not root_path:
                self._paths[name] = None
                continue
            rb_prim = _first_descendant_with_rigid_body(stage, stage.GetPrimAtPath(root_path))
            self._paths[name] = str(rb_prim.GetPath()) if rb_prim and rb_prim.IsValid() else root_path

        # One-time debug to confirm paths
        print("[Guide] Resolved paths:", self._paths)

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
        Call this once per sim tick *after* env.step() or sim.render().
        Uses a fresh XformCache so we see the latest world transforms.
        """
        idx = highlighter.step_index
        if idx >= len(self._checks):
            return

        stage: Usd.Stage = env.scene.stage
        cache = UsdGeom.XformCache()  # fresh per frame -> up-to-date if updateToUsd=True

        if self._checks[idx](env, stage, cache):
            highlighter.advance()

    # ---------------------- pose helpers ---------------------

    def _get_tf(self, stage: Usd.Stage, cache: UsdGeom.XformCache, name: str):
        p = self._paths.get(name)
        if not p:
            return None
        prim = stage.GetPrimAtPath(p)
        if not prim or not prim.IsValid():
            return None
        return cache.GetLocalToWorldTransform(prim)

    @staticmethod
    def _pos(xf): return xf.ExtractTranslation()

    @staticmethod
    def _rot(xf): return xf.ExtractRotation().GetQuat()

    # ---------------------- step checks ----------------------

    def _check_pickup_box(self, env, stage, cache) -> bool:
        table_xf = self._get_tf(stage, cache, "Table")
        box_xf   = self._get_tf(stage, cache, "DrawerBox")
        if not (table_xf and box_xf):
            if not self._warned_no_updates and box_xf is None:
                print("[Guide] WARN: Could not read live transforms. Did you set /persistent/physics/updateToUsd = True?")
                self._warned_no_updates = True
            return False
        # ≥ 6 cm above table
        return (self._pos(box_xf)[2] - self._pos(table_xf)[2]) >= 1.2

    def _check_braced_box(self, env, stage, cache) -> bool:
        obs_xf = self._get_tf(stage, cache, "ObstacleLeft")
        box_xf = self._get_tf(stage, cache, "DrawerBox")
        if not (obs_xf and box_xf):
            return False
        d = (self._pos(obs_xf) - self._pos(box_xf)).GetLength()
        ang = _ang_deg(self._rot(obs_xf), self._rot(box_xf))
        return (d <= 0.03) and (ang <= 15.0)

    def _check_bottom_insert(self, env, stage, cache) -> bool:
        box_xf = self._get_tf(stage, cache, "DrawerBox")
        bot_xf = self._get_tf(stage, cache, "DrawerBottom")
        if not (box_xf and bot_xf):
            return False
        d = (self._pos(box_xf) - self._pos(bot_xf)).GetLength()
        ang = _ang_deg(self._rot(box_xf), self._rot(bot_xf))
        dz = self._pos(bot_xf)[2] - self._pos(box_xf)[2]
        return (d <= 0.02) and (ang <= 12.0) and (dz <= -0.005)

    def _check_top_insert(self, env, stage, cache) -> bool:
        box_xf = self._get_tf(stage, cache, "DrawerBox")
        top_xf = self._get_tf(stage, cache, "DrawerTop")
        if not (box_xf and top_xf):
            return False
        d = (self._pos(box_xf) - self._pos(top_xf)).GetLength()
        ang = _ang_deg(self._rot(box_xf), self._rot(top_xf))
        dz = self._pos(top_xf)[2] - self._pos(box_xf)[2]
        return (d <= 0.02) and (ang <= 12.0) and (dz >= 0.005)
