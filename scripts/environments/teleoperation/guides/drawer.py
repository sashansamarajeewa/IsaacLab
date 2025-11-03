from .base import BaseGuide, VisualSequenceHighlighter
from pxr import UsdGeom, Usd, UsdPhysics, Gf
import math
from typing import Optional, Tuple

# Try to import PrimView from Isaac Sim Core
try:
    from omni.isaac.core.prims import RigidPrimView
    _HAS_PRIM_VIEW = True
except Exception:
    _HAS_PRIM_VIEW = False

def _ang_deg(q1: Gf.Quatd, q2: Gf.Quatd) -> float:
    dq = q1 * q2.GetInverse()
    a = 2.0 * math.degrees(math.acos(max(-1.0, min(1.0, abs(dq.GetReal())))))
    return min(a, 360.0 - a)

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

def _first_descendant_with_rigid_body(stage: Usd.Stage, root_prim: Usd.Prim) -> Optional[Usd.Prim]:
    if not root_prim or not root_prim.IsValid():
        return None
    if UsdPhysics.RigidBodyAPI(root_prim):
        return root_prim
    for p in Usd.PrimRange(root_prim):
        if UsdPhysics.RigidBodyAPI(p):
            return p
    return None

class DrawerGuide(BaseGuide):
    SEQUENCE = ["DrawerBox", "DrawerBox", "DrawerBottom", "DrawerTop"]

    def __init__(self):
        super().__init__()
        self._checks = [
            self._check_pickup_box,
            self._check_braced_box,
            self._check_bottom_insert,
            self._check_top_insert,
        ]
        self._paths: dict[str, Optional[str]] = {}
        # PrimViews for live physics poses (if available)
        self._views: dict[str, Optional["RigidPrimView"]] = {"DrawerBox": None, "DrawerBottom": None, "DrawerTop": None}

    # ------------------- lifecycle / reset -------------------

    def on_reset(self, env):
        stage: Usd.Stage = env.scene.stage
        env_ns: str = env.scene.env_ns

        # Resolve static refs normally
        for name in ("PackingTable", "ObstacleLeft", "ObstacleFront"):
            self._paths[name] = _resolve_env_scoped_path(stage, env_ns, name)

        # Resolve moving parts to rigid body prims (for USD fallback / highlighting)
        for name in ("DrawerBox", "DrawerBottom", "DrawerTop"):
            root_path = _resolve_env_scoped_path(stage, env_ns, name)
            rb_path = None
            if root_path:
                rb_prim = _first_descendant_with_rigid_body(stage, stage.GetPrimAtPath(root_path))
                rb_path = str(rb_prim.GetPath()) if rb_prim and rb_prim.IsValid() else root_path
            self._paths[name] = rb_path

        # Build PrimViews if available
        if _HAS_PRIM_VIEW:
            # Each view targets exactly one prim path
            for name in ("DrawerBox", "DrawerBottom", "DrawerTop"):
                path = self._paths.get(name)
                if not path:
                    self._views[name] = None
                    continue
                try:
                    # RigidPrimView accepts an expression; we pass the single absolute path
                    view = RigidPrimView(prim_paths_expr=[path], name=f"{name}_view", reset_xform_properties=False)
                    # Isaac Lab envs keep a PhysicsSimView at env.sim (or env.scene.physics_sim_view)
                    # RigidPrimView.initialize() will look it up automatically if not provided.
                    view.initialize()
                    self._views[name] = view
                except Exception:
                    self._views[name] = None

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
        idx = highlighter.step_index
        if idx >= len(self._checks):
            return
        stage: Usd.Stage = env.scene.stage
        cache = UsdGeom.XformCache()  # used for static refs & fallback
        if self._checks[idx](env, stage, cache):
            highlighter.advance()

    # ---------------------- pose helpers ---------------------

    def _get_live_pose(self, name: str) -> Optional[Tuple[Gf.Vec3d, Gf.Quatd]]:
        """
        Live PhysX pose via PrimView if available. Returns (pos, quat) as Gf types.
        """
        view = self._views.get(name)
        if view is None:
            return None
        try:
            # get_world_poses returns (N, 3) positions and (N, 4) quats (xyzw) per default Isaac Sim Core
            # BUT many builds return (N, 4) quats in (wxyz). We handle both.
            pos_np, quat_np = view.get_world_poses(clone=False)
            if pos_np.shape[0] < 1:
                return None
            x, y, z = float(pos_np[0, 0]), float(pos_np[0, 1]), float(pos_np[0, 2])
            # Try to detect ordering; prefer (w,x,y,z) if plausible
            w_first = abs(quat_np[0, 0]) >= max(abs(quat_np[0, 1]), abs(quat_np[0, 2]), abs(quat_np[0, 3]))
            if w_first:
                qw, qx, qy, qz = float(quat_np[0, 0]), float(quat_np[0, 1]), float(quat_np[0, 2]), float(quat_np[0, 3])
            else:  # xyzw
                qx, qy, qz, qw = float(quat_np[0, 0]), float(quat_np[0, 1]), float(quat_np[0, 2]), float(quat_np[0, 3])
            return Gf.Vec3d(x, y, z), Gf.Quatd(qw, qx, qy, qz)
        except Exception:
            return None

    def _get_pose(self, env, stage: Usd.Stage, cache: UsdGeom.XformCache, name: str) -> Optional[Tuple[Gf.Vec3d, Gf.Quatd]]:
        """
        Unified getter: try live PrimView, else USD world xform (for static refs or fallback).
        """
        # 1) PrimView (live physics)
        live = self._get_live_pose(name)
        if live is not None:
            return live

        # 2) USD fallback
        p = self._paths.get(name)
        if not p:
            return None
        prim = stage.GetPrimAtPath(p)
        if not prim or not prim.IsValid():
            return None
        xf = cache.GetLocalToWorldTransform(prim)
        return xf.ExtractTranslation(), xf.ExtractRotation().GetQuat()

    # ---------------------- step checks ---------------------

    def _check_pickup_box(self, env, stage, cache) -> bool:
        table = self._get_pose(env, stage, cache, "PackingTable")
        box   = self._get_pose(env, stage, cache, "DrawerBox")
        if not (table and box):
            return False
        table_pos, _ = table
        box_pos, _   = box
        return (box_pos[2] - table_pos[2]) >= 0.06

    def _check_braced_box(self, env, stage, cache) -> bool:
        obs = self._get_pose(env, stage, cache, "ObstacleLeft")
        box = self._get_pose(env, stage, cache, "DrawerBox")
        if not (obs and box):
            return False
        obs_pos, obs_quat = obs
        box_pos, box_quat = box
        d = (obs_pos - box_pos).GetLength()
        ang = _ang_deg(obs_quat, box_quat)
        return (d <= 0.03) and (ang <= 15.0)

    def _check_bottom_insert(self, env, stage, cache) -> bool:
        box = self._get_pose(env, stage, cache, "DrawerBox")
        bot = self._get_pose(env, stage, cache, "DrawerBottom")
        if not (box and bot):
            return False
        box_pos, box_quat = box
        bot_pos, bot_quat = bot
        d = (box_pos - bot_pos).GetLength()
        ang = _ang_deg(box_quat, bot_quat)
        dz = bot_pos[2] - box_pos[2]
        return (d <= 0.02) and (ang <= 12.0) and (dz <= -0.005)

    def _check_top_insert(self, env, stage, cache) -> bool:
        box = self._get_pose(env, stage, cache, "DrawerBox")
        top = self._get_pose(env, stage, cache, "DrawerTop")
        if not (box and top):
            return False
        box_pos, box_quat = box
        top_pos, top_quat = top
        d = (box_pos - top_pos).GetLength()
        ang = _ang_deg(box_quat, top_quat)
        dz = top_pos[2] - box_pos[2]
        return (d <= 0.02) and (ang <= 12.0) and (dz >= 0.005)
