from .base import BaseGuide, VisualSequenceHighlighter
from pxr import UsdGeom, Usd, UsdPhysics, Gf
import math

# ----------------------- math helpers -----------------------

def _ang_deg(q1: Gf.Quatd, q2: Gf.Quatd) -> float:
    """Smallest angle (deg) between two quaternions."""
    dq = q1 * q2.GetInverse()
    a = 2.0 * math.degrees(math.acos(max(-1.0, min(1.0, abs(dq.GetReal())))))
    return min(a, 360.0 - a)

# ----------------------- USD helpers ------------------------

def _first_descendant_with_rigid_body(stage: Usd.Stage, root_prim: Usd.Prim) -> Usd.Prim | None:
    """
    Return the first descendant (including root) that has UsdPhysics.RigidBodyAPI.
    If none found, return None.
    """
    if not root_prim or not root_prim.IsValid():
        return None
    # If the root itself is a rigid body, prefer it.
    if UsdPhysics.RigidBodyAPI(root_prim):
        return root_prim
    # Otherwise search children under this asset root only (bounded search).
    for p in Usd.PrimRange(root_prim):
        if UsdPhysics.RigidBodyAPI(p):
            return p
    return None

def _resolve_env_scoped_path(stage: Usd.Stage, env_root_path: str, leaf_name: str) -> str | None:
    """
    Resolve a prim path for a given leaf_name under the given env root.
    Tries exact path first; if missing, searches only within env root (no full-stage scan).
    Returns the prim path string or None.
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

class LampGuide(BaseGuide):
    """
    Multi-step flow:
      0) Pick up DrawerBox                 (highlight DrawerBox)
      1) Brace DrawerBox on ObstacleLeft   (still highlight DrawerBox)
      2) Insert DrawerBottom               (highlight DrawerBottom)
      3) Insert DrawerTop                  (highlight DrawerTop)
    """
    SEQUENCE = ["LampBase", "LampBulb", "LampHood"]

    def __init__(self):
        super().__init__()
        self._checks = [
            self._check_pickup_box,
            self._check_braced_box,
            self._check_bottom_insert,
            self._check_top_insert,
        ]
        # Cached absolute prim paths for the CURRENT env (set on reset).
        # For moving parts, these are resolved to the *rigid body* child prims.
        self._paths: dict[str, str | None] = {}

    # ------------------- lifecycle / reset -------------------

    def on_reset(self, env):
        """
        Resolve prim paths once per episode.
        For moving parts, store the rigid body child prim path so we read live physics poses.
        """
        stage: Usd.Stage = env.scene.stage
        env_ns: str = env.scene.env_ns  # e.g., "/World/envs/env_0"

        self._paths.clear()

        # Static references: resolve normally (table and obstacles typically aren't rigid bodies).
        for name in ("PackingTable", "ObstacleLeft", "ObstacleFront"):
            self._paths[name] = _resolve_env_scoped_path(stage, env_ns, name)

        # Moving parts: resolve the asset root, then find its rigid body descendant.
        for name in ("LampBase", "LampBulb", "LampHood"):
            root_path = _resolve_env_scoped_path(stage, env_ns, name)
            if not root_path:
                self._paths[name] = None
                continue
            root_prim = stage.GetPrimAtPath(root_path)
            rb_prim = _first_descendant_with_rigid_body(stage, root_prim)
            print(rb_prim)
            self._paths[name] = str(rb_prim.GetPath()) if rb_prim and rb_prim.IsValid() else root_path

    # ---------------------- HUD content ----------------------

    def step_label(self, highlighter: VisualSequenceHighlighter) -> str:
        idx, total = highlighter.step_index, (highlighter.total_steps or 1)
        return (
            f"Step 1/{total}: Pick up LampBase" if idx == 0 else
            f"Step 2/{total}: Brace LampBase against left obstacle" if idx == 1 else
            f"Step 3/{total}: Insert LampBulb into LampBase" if idx == 2 else
            f"Step 4/{total}: Insert LampHood onto LampBase" if idx == 3 else
            "Assembly complete!"
        )

    # ------------------ per-frame evaluation -----------------

    def maybe_auto_advance(self, env, highlighter: VisualSequenceHighlighter):
        """
        Called once per sim tick (AFTER env.step or sim.render).
        Creates a fresh XformCache each call so we see the latest physics poses.
        """
        idx = highlighter.step_index
        if idx >= len(self._checks):
            return

        stage: Usd.Stage = env.scene.stage
        cache = UsdGeom.XformCache()  # fresh per frame -> up-to-date world xforms

        if self._checks[idx](env, stage, cache):
            highlighter.advance()

    # ---------------------- pose helpers ---------------------

    def _get_tf(self, stage: Usd.Stage, cache: UsdGeom.XformCache, name: str):
        """Get world transform for the resolved prim name (rigid body child when applicable)."""
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
        table_xf = self._get_tf(stage, cache, "PackingTable")
        box_xf   = self._get_tf(stage, cache, "DrawerBox")
        if not (table_xf and box_xf):
            return False
        # ≥ 6 cm above table
        # print(self._pos(box_xf)[2])
        # print(self._pos(table_xf)[2])
        return (self._pos(box_xf)[2] - self._pos(table_xf)[2]) >= 1.1

    def _check_braced_box(self, env, stage, cache) -> bool:
        obs_xf = self._get_tf(stage, cache, "ObstacleLeft")
        box_xf = self._get_tf(stage, cache, "DrawerBox")
        if not (obs_xf and box_xf):
            return False
        d = (self._pos(obs_xf) - self._pos(box_xf)).GetLength()
        ang = _ang_deg(self._rot(obs_xf), self._rot(box_xf))
        # ≤ 3 cm and ≤ 15° relative yaw/roll/pitch (quat distance)
        return (d <= 0.03) and (ang <= 15.0)

    def _check_bottom_insert(self, env, stage, cache) -> bool:
        box_xf = self._get_tf(stage, cache, "DrawerBox")
        bot_xf = self._get_tf(stage, cache, "DrawerBottom")
        if not (box_xf and bot_xf):
            return False
        d = (self._pos(box_xf) - self._pos(bot_xf)).GetLength()
        ang = _ang_deg(self._rot(box_xf), self._rot(bot_xf))
        dz = self._pos(bot_xf)[2] - self._pos(box_xf)[2]
        # Near + roughly aligned + slightly below box reference (i.e., "inside")
        return (d <= 0.02) and (ang <= 12.0) and (dz <= -0.005)

    def _check_top_insert(self, env, stage, cache) -> bool:
        box_xf = self._get_tf(stage, cache, "DrawerBox")
        top_xf = self._get_tf(stage, cache, "DrawerTop")
        if not (box_xf and top_xf):
            return False
        d = (self._pos(box_xf) - self._pos(top_xf)).GetLength()
        ang = _ang_deg(self._rot(box_xf), self._rot(top_xf))
        dz = self._pos(top_xf)[2] - self._pos(box_xf)[2]
        # Near + roughly aligned + slightly above box reference (i.e., "on top")
        return (d <= 0.02) and (ang <= 12.0) and (dz >= 0.005)
