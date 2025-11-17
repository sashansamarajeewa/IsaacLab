from .base import BaseGuide, VisualSequenceHighlighter
from pxr import UsdGeom, Usd, UsdPhysics, Gf
from typing import Optional, Tuple
import math
from omni.physx import get_physx_interface
import omni.audioplayer as audioplayer

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
    
    tol_x_dbox_lo = 0.133 # distance between drawer box and left obstacle origin along X
    tol_y_dbox_fo = 0.119 # distance between drawer box and front obstacle origin along Y
    tol_z_dbox_t = 1.081 # distance between drawer box and table origin along Z
    tol_ang_dbox_fo = 180 # angle between drawer box and front obstacle origin
    tol_x_dbox_dbottom = 0.0019 # distance between drawer box and drawer bottom origin along X
    tol_y_dbox_dbottom = 0.0228 # distance between drawer box and drawer bottom origin along Y
    tol_z_dbox_dbottom = 0.0218 # distance between drawer box and drawer bottom origin along Z
    tol_ang_dbox_dbottom = 0.5 # angle between drawer box and drawer bottom origin
    tol_x_dbox_dtop = 0.0009 # distance between drawer box and drawer top origin along X
    tol_y_dbox_dtop = 0.0162 # distance between drawer box and drawer top origin along Y
    tol_z_dbox_dtop = 0.0689 # distance between drawer box and drawer top origin along Z
    tol_ang_dbox_dtop = 0.5 # angle between drawer box and drawer top origin

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
        
        self.player = audioplayer.create_audio_player()
        self.player.load_sound("/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/assembly/sound/ding.mp3")

    # ------------------- lifecycle / reset -------------------

    def on_reset(self, env):
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

    # ---------------------- HUD content ----------------------

    def step_label(self, highlighter: VisualSequenceHighlighter) -> str:
        idx, total = highlighter.step_index, (highlighter.total_steps or 1)
        if idx == 0:
            return f"Step 1/{total}: Pick up Drawer Box"
        elif idx == 1:
            return f"Step 2/{total}: Brace Drawer Box against the front and left corner obstacles"
        elif idx == 2:
            return f"Step 3/{total}: Insert Drawer Bottom into Drawer Box"
        elif idx == 3:
            return f"Step 4/{total}: Insert Drawer Top to finish"
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
        if self._checks[idx](stage):
            highlighter.advance()
            try:
                self.player.play_sound("/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/assembly/sound/ding.mp3")
            except Exception as e:
                print(f"[Audio] play failed: {e}")


    # ---------------------- live pose getters ----------------------

    def _get_live_part_pose(self, name: str, stage: Usd.Stage) -> Optional[Tuple[Gf.Vec3d, Gf.Quatd]]:
        p = self._paths.get(name)
        if not p:
            return None
        return _physx_get_pose(p)

    # ---------------------- checks ----------------------

    def _check_pickup_box(self, stage) -> bool:
        if self._static_table_pos is None:
            return False
        box_pose = self._get_live_part_pose("DrawerBox", stage)
        if not box_pose:
            return False
        box_pos, _ = box_pose
        # 1.083 meters above table
        return (box_pos[2] - self._static_table_pos[2]) >= 1.084

    def _check_braced_box(self, stage) -> bool:
        left_pos, left_quat = self._static_obstacles["ObstacleLeft"]
        front_pos, front_quat = self._static_obstacles["ObstacleFront"]
        box_pos, box_quat = self._get_live_part_pose("DrawerBox", stage)
        if not (left_pos and front_pos and box_pos):
            return False

        dx = box_pos[0] - left_pos[0]
        dy = front_pos[1] - box_pos[1]
        z_ok = (self._static_table_pos is None) or (box_pos[2] - self._static_table_pos[2]) <= self.tol_z_dbox_t
        ang_ok = _ang_deg(box_quat, front_quat) <= self.tol_ang_dbox_fo
        return (0 < dx <= self.tol_x_dbox_lo) and (0 < dy <= self.tol_y_dbox_fo) and z_ok and ang_ok

    def _check_bottom_insert(self, stage) -> bool:
        box_pos, box_quat = self._get_live_part_pose("DrawerBox", stage)
        bot_pos, bot_quat = self._get_live_part_pose("DrawerBottom", stage)
        if not (box_pos and bot_pos):
            return False

        dx = box_pos[0] - bot_pos[0]
        dy = box_pos[1] - bot_pos[1]
        dz = box_pos[2] - bot_pos[2]
        ang = _ang_deg(box_quat, bot_quat)
        return (0 < abs(dx) <= self.tol_x_dbox_dbottom) and (0 < dy <= self.tol_y_dbox_dbottom) and (0 < dz <= self.tol_z_dbox_dbottom) and (0 < ang <= self.tol_ang_dbox_dbottom)

    def _check_top_insert(self, stage) -> bool:
        box_pos, box_quat = self._get_live_part_pose("DrawerBox", stage)
        top_pos, top_quat = self._get_live_part_pose("DrawerTop", stage)
        if not (box_pos and top_pos):
            return False

        dx = top_pos[0] - box_pos[0]
        print("dx:")
        print(dx)
        dy = box_pos[1] - top_pos[1]
        print("dy:")
        print(dy)
        dz = top_pos[2] - box_pos[2]
        print("dz:")
        print(dz)
        ang = _ang_deg(box_quat, top_quat)
        print("ang:")
        print(ang)
        return (0 < abs(dx) <= self.tol_x_dbox_dtop) and (0 < dy <= self.tol_y_dbox_dtop) and (0 < dz <= self.tol_z_dbox_dtop) and (0 < ang <= self.tol_ang_dbox_dtop)
