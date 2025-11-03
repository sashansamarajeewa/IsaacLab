from .base import BaseGuide, VisualSequenceHighlighter
from pxr import UsdGeom, Usd, UsdPhysics, Gf
import math
from typing import Optional, Tuple, Dict, Any

# ----------------------- math helpers -----------------------

def _ang_deg(q1: Gf.Quatd, q2: Gf.Quatd) -> float:
    """Smallest angle (deg) between two quaternions."""
    dq = q1 * q2.GetInverse()
    a = 2.0 * math.degrees(math.acos(max(-1.0, min(1.0, abs(dq.GetReal())))))
    return min(a, 360.0 - a)

# ----------------------- USD helpers ------------------------

def _first_descendant_with_rigid_body(stage: Usd.Stage, root_prim: Usd.Prim) -> Optional[Usd.Prim]:
    """
    Return the first descendant (including root) that has UsdPhysics.RigidBodyAPI.
    If none found, return None.
    """
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

# --------------------- pose conversions ---------------------

def _gf_from_tensor_pos_quat(pos_tensor, quat_tensor) -> Tuple[Gf.Vec3d, Gf.Quatd]:
    """
    Convert Isaac Lab tensors to Gf types.
    pos_tensor: (3,) [x,y,z]; quat_tensor: (4,) [qw,qx,qy,qz]
    """
    x, y, z = float(pos_tensor[0]), float(pos_tensor[1]), float(pos_tensor[2])
    qw, qx, qy, qz = float(quat_tensor[0]), float(quat_tensor[1]), float(quat_tensor[2]), float(quat_tensor[3])
    return Gf.Vec3d(x, y, z), Gf.Quatd(qw, qx, qy, qz)

# -------------------- scene object lookup -------------------

def _iter_scene_objects(env) -> Dict[str, Any]:
    """
    Try to expose a dict-like view of scene objects across common Isaac Lab shapes.
    Returns a mapping {registry_key: object}.
    """
    # Many scene implementations expose one of these:
    for attr in ("object_registry", "objects", "prims", "entities"):
        reg = getattr(env.scene, attr, None)
        if isinstance(reg, dict) and reg:
            return reg
    # Some expose list-like 'objects'
    objs = getattr(env.scene, "objects", None)
    if isinstance(objs, (list, tuple)) and objs:
        return {getattr(o, "name", f"obj_{i}"): o for i, o in enumerate(objs)}
    return {}

def _find_scene_object_by_leaf(env, env_ns: str, leaf: str):
    """
    Find a scene object whose prim path ends with f\"{env_ns}/{leaf}\" or whose name equals leaf.
    Returns the object or None.
    """
    # 1) Direct get_object if available
    get_object = getattr(env.scene, "get_object", None)
    if callable(get_object):
        try:
            obj = get_object(leaf)
            if obj is not None:
                return obj
        except Exception:
            pass

    # 2) Search registries / lists
    reg = _iter_scene_objects(env)
    suffix = f"{env_ns}/{leaf}"
    for key, obj in reg.items():
        prim_path = getattr(obj, "prim_path", "") or getattr(obj, "root_prim_path", "")
        name = getattr(obj, "name", "")
        if name == leaf:
            return obj
        if isinstance(prim_path, str) and prim_path.endswith(suffix):
            return obj

    # 3) Nothing found
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
        # Cached absolute prim paths for the CURRENT env (set on reset).
        # For moving parts, these may be resolved to rigid body child prims.
        self._paths: dict[str, Optional[str]] = {}
        # Isaac Lab object handles (preferred for live physics pose)
        self._objs: dict[str, Optional[object]] = {}
        # Single-env index we read from (extend if you later vectorize)
        self._env_index: int = 0
        # Debug toggle
        self._debug_print_once = True

    # ------------------- lifecycle / reset -------------------

    def on_reset(self, env):
        """
        Resolve prim paths and Isaac Lab object handles once per episode.
        Prefer Isaac Lab tensors for live pose; otherwise fall back to USD.
        """
        stage: Usd.Stage = env.scene.stage
        env_ns: str = env.scene.env_ns  # e.g., "/World/envs/env_0"

        self._paths.clear()
        self._objs.clear()

        # Try to bind Isaac Lab objects (best for live simulation poses).
        for name in ("DrawerBox", "DrawerBottom", "DrawerTop"):
            obj = _find_scene_object_by_leaf(env, env_ns, name)
            # Accept only if it exposes tensor state
            if obj is not None and hasattr(obj, "data") and hasattr(obj.data, "root_state_w"):
                self._objs[name] = obj
            else:
                self._objs[name] = None

        # Static references (table/obstacles): resolve normally.
        for name in ("Table", "ObstacleLeft", "ObstacleFront"):
            self._paths[name] = _resolve_env_scoped_path(stage, env_ns, name)

        # Moving parts: resolve to rigid body prim (fallback path if rigid not found).
        for name in ("DrawerBox", "DrawerBottom", "DrawerTop"):
            root_path = _resolve_env_scoped_path(stage, env_ns, name)
            if not root_path:
                self._paths[name] = None
                continue
            root_prim = stage.GetPrimAtPath(root_path)
            rb_prim = _first_descendant_with_rigid_body(stage, root_prim)
            self._paths[name] = str(rb_prim.GetPath()) if rb_prim and rb_prim.IsValid() else root_path

        # If you ever run multiple envs, set self._env_index accordingly per env instance.
        self._env_index = 0

        # Optional one-time debug
        if self._debug_print_once:
            try:
                reg = _iter_scene_objects(env)
                print("[Guide] Scene registry keys:", list(reg.keys())[:16], "...")
                print("[Guide] Tensor-bound objects:", {k: (v is not None) for k, v in self._objs.items()})
                print("[Guide] Resolved USD paths:", self._paths)
            except Exception:
                pass
            self._debug_print_once = False

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
        Called once per sim tick (AFTER env.step or sim.render).
        Creates a fresh XformCache each call for USD fallback so we see the latest physics poses.
        """
        idx = highlighter.step_index
        if idx >= len(self._checks):
            return

        stage: Usd.Stage = env.scene.stage
        cache = UsdGeom.XformCache()  # fresh per frame (USD fallback path)

        if self._checks[idx](env, stage, cache):
            highlighter.advance()

    # ---------------------- pose helpers ---------------------

    def _get_pose(self, env, stage: Usd.Stage, cache: UsdGeom.XformCache, name: str) -> Optional[Tuple[Gf.Vec3d, Gf.Quatd]]:
        """
        Return (pos, quat) as Gf types for the named part.
        Prefers Isaac Lab object tensors; falls back to USD world xform.
        """
        # 1) Isaac Lab tensor path (preferred)
        obj = self._objs.get(name)
        if obj is not None:
            try:
                rs = obj.data.root_state_w  # [num_envs, 13]
                pos = rs[self._env_index, 0:3]
                quat = rs[self._env_index, 3:7]  # [qw,qx,qy,qz]
                return _gf_from_tensor_pos_quat(pos, quat)
            except Exception:
                # fall through to USD if tensor not available for some reason
                pass

        # 2) USD fallback path
        p = self._paths.get(name)
        if not p:
            return None
        prim = stage.GetPrimAtPath(p)
        if not prim or not prim.IsValid():
            return None
        xf = cache.GetLocalToWorldTransform(prim)
        pos = xf.ExtractTranslation()
        quat = xf.ExtractRotation().GetQuat()
        return pos, quat

    # ---------------------- step checks ---------------------

    def _check_pickup_box(self, env, stage, cache) -> bool:
        table_pose = self._get_pose(env, stage, cache, "Table")
        box_pose   = self._get_pose(env, stage, cache, "DrawerBox")
        if not (table_pose and box_pose):
            return False
        table_pos, _ = table_pose
        box_pos, _   = box_pose
        # ≥ 6 cm above table
        return (box_pos[2] - table_pos[2]) >= 1.2

    def _check_braced_box(self, env, stage, cache) -> bool:
        obs_pose = self._get_pose(env, stage, cache, "ObstacleLeft")
        box_pose = self._get_pose(env, stage, cache, "DrawerBox")
        if not (obs_pose and box_pose):
            return False
        obs_pos, obs_quat = obs_pose
        box_pos, box_quat = box_pose
        d = (obs_pos - box_pos).GetLength()
        ang = _ang_deg(obs_quat, box_quat)
        # ≤ 3 cm and ≤ 15° relative orientation
        return (d <= 0.03) and (ang <= 15.0)

    def _check_bottom_insert(self, env, stage, cache) -> bool:
        box_pose = self._get_pose(env, stage, cache, "DrawerBox")
        bot_pose = self._get_pose(env, stage, cache, "DrawerBottom")
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
        box_pose = self._get_pose(env, stage, cache, "DrawerBox")
        top_pose = self._get_pose(env, stage, cache, "DrawerTop")
        if not (box_pose and top_pose):
            return False
        box_pos, box_quat = box_pose
        top_pos, top_quat = top_pose
        d = (box_pos - top_pos).GetLength()
        ang = _ang_deg(box_quat, top_quat)
        dz = top_pos[2] - box_pos[2]
        # Near + roughly aligned + slightly above box reference (i.e., "on top")
        return (d <= 0.02) and (ang <= 12.0) and (dz >= 0.005)
