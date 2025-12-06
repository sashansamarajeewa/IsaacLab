from .base import BaseGuide, VisualSequenceHighlighter, ang_deg, first_descendant_with_rigid_body, resolve_env_scoped_path, spawn_ghost_preview, MaterialRegistry
from pxr import UsdGeom, Usd, Gf
from typing import Optional, Tuple

# ======================= Drawer Guide =======================

class HexagonGuide(BaseGuide):

    SEQUENCE = ["part0", "part1", "part2", "part3", "part4", "part5", "part6"]
    MOVING_PARTS = ["part0", "part1", "part2", "part3", "part4", "part5", "part6"]
    STATIC_PARTS = ("ObstacleLeft", "ObstacleFront", "ObstacleRight")
    
    # tol_x_dbox_lo = 0.133 # distance between drawer box and left obstacle origin along X
    # tol_y_dbox_fo = 0.119 # distance between drawer box and front obstacle origin along Y
    # tol_ang_dbox_fo = 180 # angle between drawer box and front obstacle origin
    # tol_x_dbox_dbottom = 0.0019 # distance between drawer box and drawer bottom origin along X
    # tol_y_dbox_dbottom = 0.0228 # distance between drawer box and drawer bottom origin along Y
    # tol_z_dbox_dbottom = 0.0218 # distance between drawer box and drawer bottom origin along Z
    # tol_ang_dbox_dbottom = 0.5 # angle between drawer box and drawer bottom origin
    # tol_x_dbox_dtop = 0.0010 # distance between drawer box and drawer top origin along X
    # tol_y_dbox_dtop = 0.0162 # distance between drawer box and drawer top origin along Y
    # tol_z_dbox_dtop = 0.0689 # distance between drawer box and drawer top origin along Z
    # tol_ang_dbox_dtop = 0.5 # angle between drawer box and drawer top origin
    tol_z_dbox_t = 1.084 # distance between drawer box and table origin along Z

    tgt_box_pos = Gf.Vec3d(-0.23737475275993347, 0.5523679852485657, 1.0766514539718628)
    tgt_box_quat = Gf.Quatd(-7.106468547135592e-05, Gf.Vec3d(7.105479744495824e-05, -0.7071069478988647, 0.7071067690849304))
    tgt_bot_pos = Gf.Vec3d(-0.23738756775856018, 0.5294921398162842, 1.0552730560302734)
    tgt_bot_quat = Gf.Quatd(9.714877523947507e-05, Gf.Vec3d(-0.000178157992195338, -0.7075424790382385, 0.7066707611083984))
    tgt_top_pos = Gf.Vec3d(-0.23674903810024261, 0.5366296172142029, 1.1454159021377563)
    tgt_top_quat = Gf.Quatd(-0.0023043914698064327, Gf.Vec3d(0.0004399800091050565, -0.7075466513633728, 0.7066628336906433))

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
            "part0": None,
            "part1": None,
            "part2": None,
            "part3": None,
            "part4": None,
            "part5": None,
            "part6": None,
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
            "part0": None,
            "part1": None,
            "part2": None,
            "part3": None,
            "part4": None,
            "part5": None,
            "part6": None,
        }
        
        # Ghost prim paths by logical name
        self._ghost_paths_by_name: dict[str, str] = {}

    # ------------------- reset -------------------

    def on_reset(self, env):
        super().on_reset(env)
        stage: Usd.Stage = env.scene.stage
        env_ns: str = env.scene.env_ns
        self._paths.clear()
        self._asset_roots = {"part0": None,"part1": None,"part2": None,"part3": None,"part4": None, "part5": None,"part6": None}
        self._target_poses = {"part0": None,"part1": None,"part2": None,"part3": None,"part4": None, "part5": None,"part6": None}
        self._ghost_paths_by_name.clear()
        self._static_table_pos = None
        self._static_obstacles = {"ObstacleLeft": None, "ObstacleFront": None, "ObstacleRight": None}

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
            rb_prim = first_descendant_with_rigid_body(stage, stage.GetPrimAtPath(root_path))
            self._paths[name] = str(rb_prim.GetPath()) if rb_prim and rb_prim.IsValid() else root_path

        # Cache static world poses once
        cache = UsdGeom.XformCache()
        if self._paths.get("Table"):
            prim = stage.GetPrimAtPath(self._paths["Table"])
            if prim and prim.IsValid():
                self._static_table_pos = cache.GetLocalToWorldTransform(prim).ExtractTranslation()

        for name in self.STATIC_PARTS:
            p = self._paths.get(name)
            if not p:
                continue
            prim = stage.GetPrimAtPath(p)
            if prim and prim.IsValid():
                xf = cache.GetLocalToWorldTransform(prim)
                self._static_obstacles[name] = (xf.ExtractTranslation(), xf.ExtractRotation().GetQuat())

        # --------- Compute simple target poses for previews ---------
        if (
            self._static_table_pos is not None
            and self._static_obstacles["ObstacleLeft"] is not None
            and self._static_obstacles["ObstacleFront"] is not None
        ):

            self._target_poses["part0"] = (self.tgt_box_pos, self.tgt_box_quat)

            self._target_poses["part1"] = (self.tgt_bot_pos, self.tgt_bot_quat)

            self._target_poses["part2"] = (self.tgt_top_pos, self.tgt_top_quat)

            self._target_poses["part3"] = (self.tgt_top_pos, self.tgt_top_quat)

            self._target_poses["part4"] = (self.tgt_top_pos, self.tgt_top_quat)

            self._target_poses["part5"] = (self.tgt_top_pos, self.tgt_top_quat)

            self._target_poses["part6"] = (self.tgt_top_pos, self.tgt_top_quat)

        # --------- Spawn/update ghosts at target poses ---------
        stage = self._stage  # cached from super().on_reset
        if stage is not None:
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
        box_pose = self.get_live_part_pose("part0")
        if not box_pose:
            return False
        box_pos, _ = box_pose
        # 1.084 meters above table
        print(self.get_live_part_pose("part0"))
        print(self.get_live_part_pose("part1"))
        print(self.get_live_part_pose("part2"))
        print(self.get_live_part_pose("part3"))
        print(self.get_live_part_pose("part4"))
        print(self.get_live_part_pose("part5"))
        print(self.get_live_part_pose("part6"))
        print("##############")
        return (box_pos[0] - self._static_table_pos[2]) >= self.tol_z_dbox_t

    def _check_braced_box(self) -> bool:
        tgt = self._target_poses.get("DrawerBox")
        live = self.get_live_part_pose("DrawerBox")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)
        
        return pos_err <= 0.01 and ang_err <= 3.0

    def _check_bottom_insert(self) -> bool:
        tgt = self._target_poses.get("DrawerBottom")
        live = self.get_live_part_pose("DrawerBottom")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)
        
        return pos_err <= 0.01 and ang_err <= 3.0

    def _check_top_insert(self) -> bool:
        tgt = self._target_poses.get("DrawerTop")
        live = self.get_live_part_pose("DrawerTop")
        if not (tgt and live):
            return False

        live_pos, live_quat = live
        tgt_pos, tgt_quat = tgt
        pos_err = (live_pos - tgt_pos).GetLength()
        ang_err = ang_deg(live_quat, tgt_quat)
        
        return pos_err <= 0.01 and ang_err <= 3.0
