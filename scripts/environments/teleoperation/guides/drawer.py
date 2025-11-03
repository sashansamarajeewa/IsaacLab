from .base import BaseGuide, VisualSequenceHighlighter
from pxr import UsdGeom, Usd, UsdPhysics, Gf
import math
from typing import Optional, Tuple

# Isaac Core PrimView (live PhysX poses)
try:
    from omni.isaac.core.prims import RigidPrimView
    _HAS_PRIM_VIEW = True
except Exception:
    _HAS_PRIM_VIEW = False

# ----------------------- math helpers -----------------------

def _ang_deg(q1: Gf.Quatd, q2: Gf.Quatd) -> float:
    dq = q1 * q2.GetInverse()
    a = 2.0 * math.degrees(math.acos(max(-1.0, min(1.0, abs(dq.GetReal())))))
    return min(a, 360.0 - a)

# ----------------------- USD helpers ------------------------

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

# ======================= Drawer Guide =======================

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
        self._views: dict[str, Optional["RigidPrimView"]] = {"DrawerBox": None, "DrawerBottom": None, "DrawerTop": None}
        self._debug_once = True

    # ------------------- lifecycle / reset -------------------

    def on_reset(self, env):
        stage: Usd.Stage = env.scene.stage
        env_ns: str = getattr(env.scene, "env_ns", "/World/envs/env_0")

        # Resolve static refs (table/obstacles)
        for name in ("PackingTable", "ObstacleLeft", "ObstacleFront"):
            self._paths[name] = _resolve_env_scoped_path(stage, env_ns, name)

        # Resolve moving parts to a rigid body prim path
        for name in ("DrawerBox", "DrawerBottom", "DrawerTop"):
            root_path = _resolve_env_scoped_path(stage, env_ns, name)
            if not root_path:
                self._paths[name] = None
                continue
            rb_prim = _first_descendant_with_rigid_body(stage, stage.GetPrimAtPath(root_path))
            self._paths[name] = str(rb_prim.GetPath()) if rb_prim and rb_prim.IsValid() else root_path

        # Build PrimViews robustly
        for name in ("DrawerBox", "DrawerBottom", "DrawerTop"):
            self._views[name] = self._make_view(env, self._paths.get(name), name)

        if self._debug_once:
            print("[Guide] PrimView support:", _HAS_PRIM_VIEW)
            print("[Guide] Prim paths:", self._paths)
            for n, v in self._views.items():
                msg = "None"
                if v is not None:
                    try:
                        msg = f"valid={v.is_valid()}, num_prims={getattr(v, 'count', getattr(v, 'num_prims', '??'))}"
                    except Exception:
                        msg = "constructed"
                print(f"[Guide] View[{n}]: {msg}")
            self._debug_once = False

    def _make_view(self, env, path: Optional[str], name: str) -> Optional["RigidPrimView"]:
        if not _HAS_PRIM_VIEW or not path:
            return None
        try:
            # Get the physics sim view from env/scene (covers multiple Isaac versions)
            phys_view = getattr(env.scene, "physics_sim_view", None)
            if phys_view is None:
                sim = getattr(env, "sim", None)
                # Isaac Lab usually exposes .physics_sim_view on env.sim
                phys_view = getattr(sim, "physics_sim_view", None)
                # Some builds expose a getter
                if phys_view is None and hasattr(sim, "get_physics_sim_view"):
                    phys_view = sim.get_physics_sim_view()

            view = RigidPrimView(prim_paths_expr=[path], name=f"{name}_view", reset_xform_properties=False)
            # Initialize with explicit physics sim view when available
            if phys_view is not None:
                view.initialize(physics_sim_view=phys_view)
            else:
                view.initialize()

            # Validate the view actually bound at least one rigid
            num = getattr(view, "count", None)
            if num is None:
                num = getattr(view, "num_prims", None)
            if num is None:
                # Last resort: try to query poses to check binding
                try:
                    pos, _ = view.get_world_poses(clone=True)
                    num = pos.shape[0]
                except Exception:
                    num = 0
            if num < 1:
                return None
            return view
        except Exception as e:
            print(f"[Guide] Failed to build PrimView for {name} @ {path}: {e}")
            return None

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
        # IMPORTANT: advance sim first in your loop; then we read here.
        stage: Usd.Stage = env.scene.stage
        cache = UsdGeom.XformCache()  # used for static refs & USD fallback
        # Ensure PrimViews pull latest buffers (older APIs)
        for v in self._views.values():
            if v is not None:
                try:
                    # Some versions require update(dt); others refresh internally on get_world_poses()
                    v.update(dt=0.0)
                except Exception:
                    pass
        if self._checks[idx](env, stage, cache):
            highlighter.advance()

    # ---------------------- pose helpers ---------------------

    def _get_live_pose(self, name: str) -> Optional[Tuple[Gf.Vec3d, Gf.Quatd]]:
        view = self._views.get(name)
        if view is None:
            return None
        try:
            pos_np, quat_np = view.get_world_poses(clone=False)
            if pos_np is None or pos_np.shape[0] < 1:
                return None
            x, y, z = float(pos_np[0, 0]), float(pos_np[0, 1]), float(pos_np[0, 2])
            # Heuristic for quat order
            w_first = abs(quat_np[0, 0]) >= max(abs(quat_np[0, 1]), abs(quat_np[0, 2]), abs(quat_np[0, 3]))
            if w_first:
                qw, qx, qy, qz = float(quat_np[0, 0]), float(quat_np[0, 1]), float(quat_np[0, 2]), float(quat_np[0, 3])
            else:
                qx, qy, qz, qw = float(quat_np[0, 0]), float(quat_np[0, 1]), float(quat_np[0, 2]), float(quat_np[0, 3])
            return Gf.Vec3d(x, y, z), Gf.Quatd(qw, qx, qy, qz)
        except Exception:
            return None

    def _get_pose(self, env, stage: Usd.Stage, cache: UsdGeom.XformCache, name: str) -> Optional[Tuple[Gf.Vec3d, Gf.Quatd]]:
        # 1) Live PhysX via PrimView
        live = self._get_live_pose(name)
        if live is not None:
            return live
        # 2) USD fallback (requires updateToUsd=True to be live)
        p = self._paths.get(name)
        if not p:
            return None
        prim = stage.GetPrimAtPath(p)
        if not prim or not prim.IsValid():
            return None
        xf = cache.GetLocalToWorldTransform(prim)
        return xf.ExtractTranslation(), xf.ExtractRotation().GetQuat()

    # ---------------------- step checks ----------------------

    def _check_pickup_box(self, env, stage, cache) -> bool:
        table = self._get_pose(env, stage, cache, "PackingTable")
        box   = self._get_pose(env, stage, cache, "DrawerBox")
        if not (table and box):
            return False
        table_pos, _ = table
        box_pos, _   = box
        return (box_pos[2] - table_pos[2]) >= 1.2

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
