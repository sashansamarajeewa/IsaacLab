from .base import BaseGuide, VisualSequenceHighlighter, ang_deg, first_descendant_with_rigid_body, resolve_env_scoped_path, spawn_ghost_preview, MaterialRegistry
from pxr import UsdGeom, Usd, UsdPhysics, Gf
from typing import Optional, Tuple

# ======================= Drawer Guide =======================

class DrawerGuide(BaseGuide):

    SEQUENCE = ["DrawerBox", "DrawerBox", "DrawerBottom", "DrawerTop"]
    
    tol_x_dbox_lo = 0.133 # distance between drawer box and left obstacle origin along X
    tol_y_dbox_fo = 0.119 # distance between drawer box and front obstacle origin along Y
    tol_z_dbox_t = 1.081 # distance between drawer box and table origin along Z
    tol_ang_dbox_fo = 180 # angle between drawer box and front obstacle origin
    tol_x_dbox_dbottom = 0.0019 # distance between drawer box and drawer bottom origin along X
    tol_y_dbox_dbottom = 0.0228 # distance between drawer box and drawer bottom origin along Y
    tol_z_dbox_dbottom = 0.0218 # distance between drawer box and drawer bottom origin along Z
    tol_ang_dbox_dbottom = 0.5 # angle between drawer box and drawer bottom origin
    tol_x_dbox_dtop = 0.0010 # distance between drawer box and drawer top origin along X
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
        # Resolved prim paths. Moving parts - rigid body prim if available
        self._paths: dict[str, Optional[str]] = {}
        # Asset root paths for ghosts
        self._asset_roots: dict[str, Optional[str]] = {
            "DrawerBox": None,
            "DrawerBottom": None,
            "DrawerTop": None,
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
            "DrawerBox": None,
            "DrawerBottom": None,
            "DrawerTop": None,
        }
        
        # Ghost prim paths by logical name
        self._ghost_paths_by_name: dict[str, str] = {}

    # ------------------- reset -------------------

    def on_reset(self, env):
        super().on_reset(env)
        stage: Usd.Stage = env.scene.stage
        env_ns: str = env.scene.env_ns
        self._paths.clear()
        self._asset_roots = {"DrawerBox": None, "DrawerBottom": None, "DrawerTop": None}
        self._target_poses = {"DrawerBox": None, "DrawerBottom": None, "DrawerTop": None}
        self._ghost_paths_by_name.clear()
        self._static_table_pos = None
        self._static_obstacles = {"ObstacleLeft": None, "ObstacleFront": None, "ObstacleRight": None}

        # Table (static)
        table_path = resolve_env_scoped_path(stage, env_ns, "PackingTable")
        self._paths["Table"] = table_path

        # Obstacles (static)
        for name in ("ObstacleLeft", "ObstacleFront", "ObstacleRight"):
            self._paths[name] = resolve_env_scoped_path(stage, env_ns, name)

        # Moving parts - rigid body prim if present else root
        for name in ("DrawerBox", "DrawerBottom", "DrawerTop"):
            root_path = resolve_env_scoped_path(stage, env_ns, name)
            self._asset_roots[name] = root_path
            if not root_path:
                self._paths[name] = None
                continue
            rb_prim = first_descendant_with_rigid_body(stage, stage.GetPrimAtPath(root_path))
            self._paths[name] = str(rb_prim.GetPath()) if rb_prim and rb_prim.IsValid() else root_path

        # Cache static world poses once
        cache = UsdGeom.XformCache()
        if self._paths.get("Table"):
            prim = stage.GetPrimAtPath(self._paths["Table"])
            if prim and prim.IsValid():
                self._static_table_pos = cache.GetLocalToWorldTransform(prim).ExtractTranslation()

        for name in ("ObstacleLeft", "ObstacleFront", "ObstacleRight"):
            p = self._paths.get(name)
            if not p:
                continue
            prim = stage.GetPrimAtPath(p)
            if prim and prim.IsValid():
                xf = cache.GetLocalToWorldTransform(prim)
                self._static_obstacles[name] = (xf.ExtractTranslation(), xf.ExtractRotation().GetQuat())

        # --------- Compute simple target poses for previews ---------
        # NOTE: these are approximate, based on your tolerances.
        if (
            self._static_table_pos is not None
            and self._static_obstacles["ObstacleLeft"] is not None
            and self._static_obstacles["ObstacleFront"] is not None
        ):
            left_pos, left_quat = self._static_obstacles["ObstacleLeft"]
            front_pos, front_quat = self._static_obstacles["ObstacleFront"]
            table_z = self._static_table_pos[2]

            # target DrawerBox braced in corner (roughly within tolerance bounds)
            box_pos = Gf.Vec3d(
                left_pos[0] + self.tol_x_dbox_lo,
                front_pos[1] - self.tol_y_dbox_fo,
                table_z + self.tol_z_dbox_t,
            )
            box_quat = 180+front_quat
            self._target_poses["DrawerBox"] = (box_pos, box_quat)

            # DrawerBottom: relative to box target
            bot_pos = Gf.Vec3d(
                box_pos[0],
                box_pos[1] - self.tol_y_dbox_dbottom,
                box_pos[2] - self.tol_z_dbox_dbottom,
            )
            bot_quat = box_quat
            self._target_poses["DrawerBottom"] = (bot_pos, bot_quat)

            # DrawerTop: relative to box target
            top_pos = Gf.Vec3d(
                box_pos[0],
                box_pos[1] - self.tol_y_dbox_dtop,
                box_pos[2] + self.tol_z_dbox_dtop,
            )
            top_quat = box_quat
            self._target_poses["DrawerTop"] = (top_pos, top_quat)

        # --------- Spawn/update ghosts at target poses ---------
        stage = self._stage  # cached from super().on_reset
        if stage is not None:
            for name in ("DrawerBox", "DrawerBottom", "DrawerTop"):
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
        self._update_ghost_visibility_for_step(0)
    
    def get_all_instructions(self) -> list[str]:
        total = len(self.SEQUENCE)
        base_steps = [
        f"Step 1/{total}: Pick up Drawer Box",
        f"Step 2/{total}: Brace Drawer Box against the front and left corner obstacles",
        f"Step 3/{total}: Insert Drawer Bottom into Drawer Box",
        f"Step 4/{total}: Insert Drawer Top to finish",
        ]
        base_steps.append("Assembly complete! Press 'Stop' button")
        return base_steps
    
    # ---------------------- HUD content ----------------------

    # def step_label(self, highlighter: VisualSequenceHighlighter) -> str:
    #     idx, total = highlighter.step_index, (highlighter.total_steps or 1)
    #     if idx == 0:
    #         return f"Step 1/{total}: Pick up Drawer Box"
    #     elif idx == 1:
    #         return f"Step 2/{total}: Brace Drawer Box against the front and left corner obstacles"
    #     elif idx == 2:
    #         return f"Step 3/{total}: Insert Drawer Bottom into Drawer Box"
    #     elif idx == 3:
    #         return f"Step 4/{total}: Insert Drawer Top to finish"
    #     return "Assembly complete!"

    # ---------------------- checks ----------------------

    def _check_pickup_box(self) -> bool:
        if self._static_table_pos is None:
            return False
        box_pose = self.get_live_part_pose("DrawerBox")
        if not box_pose:
            return False
        box_pos, _ = box_pose
        # 1.084 meters above table
        return (box_pos[2] - self._static_table_pos[2]) >= 1.084

    def _check_braced_box(self) -> bool:
        left_pos, left_quat = self._static_obstacles["ObstacleLeft"]
        front_pos, front_quat = self._static_obstacles["ObstacleFront"]
        box_pos, box_quat = self.get_live_part_pose("DrawerBox")
        if not (left_pos and front_pos and box_pos):
            return False

        dx = box_pos[0] - left_pos[0]
        dy = front_pos[1] - box_pos[1]
        z_ok = (self._static_table_pos is None) or (box_pos[2] - self._static_table_pos[2]) <= self.tol_z_dbox_t
        ang_ok = ang_deg(box_quat, front_quat) <= self.tol_ang_dbox_fo
        return (0 < dx <= self.tol_x_dbox_lo) and (0 < dy <= self.tol_y_dbox_fo) and z_ok and ang_ok

    def _check_bottom_insert(self) -> bool:
        box_pos, box_quat = self.get_live_part_pose("DrawerBox")
        bot_pos, bot_quat = self.get_live_part_pose("DrawerBottom")
        if not (box_pos and bot_pos):
            return False

        dx = box_pos[0] - bot_pos[0]
        dy = box_pos[1] - bot_pos[1]
        dz = box_pos[2] - bot_pos[2]
        ang = ang_deg(box_quat, bot_quat)
        return (0 < abs(dx) <= self.tol_x_dbox_dbottom) and (0 < dy <= self.tol_y_dbox_dbottom) and (0 < dz <= self.tol_z_dbox_dbottom) and (0 < ang <= self.tol_ang_dbox_dbottom)

    def _check_top_insert(self) -> bool:
        box_pos, box_quat = self.get_live_part_pose("DrawerBox")
        top_pos, top_quat = self.get_live_part_pose("DrawerTop")
        if not (box_pos and top_pos):
            return False

        dx = top_pos[0] - box_pos[0]
        dy = box_pos[1] - top_pos[1]
        dz = top_pos[2] - box_pos[2]
        ang = ang_deg(box_quat, top_quat)
        return (0 < abs(dx) <= self.tol_x_dbox_dtop) and (0 < dy <= self.tol_y_dbox_dtop) and (0 < dz <= self.tol_z_dbox_dtop) and (0 < ang <= self.tol_ang_dbox_dtop)
