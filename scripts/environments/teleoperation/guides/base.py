from __future__ import annotations
import textwrap
from typing import Callable, List, Optional, Sequence, Tuple
from pxr import Usd, Gf, UsdPhysics, UsdGeom
from isaaclab.sim import utils as sim_utils
from isaaclab.sim.spawners.materials import spawn_preview_surface, spawn_rigid_body_material, spawn_from_mdl_file
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg, GlassMdlCfg
from pxr import UsdShade
from omni.physx import get_physx_interface
import math

# ---------------------------------------------------------------------------
# Material registry: define & spawn once, then bind everywhere
# ---------------------------------------------------------------------------

class MaterialRegistry:
    """
    Owns a simple visual highlight material and a physics grasp material.
    Call ensure_all(stage) once (e.g., before first reset).
    """
    visual_path = "/World/Materials/Highlight"
    physics_path = "/World/Materials/Grasp"
    ghost_path   = "/World/Materials/GhostPreview"

    # Visual highlighter material (preview surface)
    visual_cfg = PreviewSurfaceCfg(
        diffuse_color=(0.6, 0.8, 0.1),
        emissive_color=(0.0, 0.0, 0.0),
    )

    # Physics (friction) material
    physics_cfg = RigidBodyMaterialCfg(
        static_friction=0.8,
        dynamic_friction=0.7,
        friction_combine_mode="multiply",
    )
    
    # Ghost preview material
    ghost_cfg = GlassMdlCfg(
        glass_color=(0.871, 1, 0.957),
        frosting_roughness=0.2,
        thin_walled=True,
    )

    @classmethod
    def ensure_all(cls, stage: Usd.Stage) -> None:
        # Create/update visual material
        spawn_preview_surface(prim_path=cls.visual_path, cfg=cls.visual_cfg)
        # Create/update physics material
        spawn_rigid_body_material(prim_path=cls.physics_path, cfg=cls.physics_cfg)
        # Create/update host preview material
        spawn_from_mdl_file(prim_path=cls.ghost_path, cfg=cls.ghost_cfg)

# ---------------------------------------------------------------------------
# Visual highlighter using bind_visual_material at the asset root
# ---------------------------------------------------------------------------

class VisualSequenceHighlighter:
    """
    Highlights assets by *leaf name* using bind_visual_material at the asset root.
    - Provide a SEQUENCE of names.
    - Call refresh_after_reset() after env resets.
    """
    def __init__(self, stage: Usd.Stage, sequence: List[str], visual_mat_path: str):
        self._stage = stage
        self._seq = sequence[:] if sequence else []
        self._mat_path = visual_mat_path
        self._step = 0
        self._active_visuals_paths: list[str] = []
    
    def _unbind_all(self):
        for p in self._active_visuals_paths:
            # safe: only unbind where we bound before
            UsdShade.MaterialBindingAPI(self._stage.GetPrimAtPath(p)).UnbindDirectBinding()
        self._active_visuals_paths = []

    def _bind_name(self, leaf_name: Optional[str]):
        self._unbind_all()
        if not leaf_name:
            return

        # resolve all asset roots for this leaf across envs
        asset_roots = _find_asset_roots_by_leaf_name(leaf_name)

        # map each root to its 'visuals' path; if missing, fall back to root
        targets = []
        for root_path in asset_roots:
            vpath = _find_visuals_path(self._stage, root_path)
            targets.append(vpath if vpath else root_path)

        # de-dupe while preserving order
        seen, uniq = set(), []
        for p in targets:
            if p not in seen:
                uniq.append(p); seen.add(p)

        # bind only on visuals
        for p in uniq:
            sim_utils.bind_visual_material(p, self._mat_path, stronger_than_descendants=True)

        self._active_visuals_paths = uniq

    def set_sequence(self, names: List[str], start_index: int = 0):
        self._seq = names[:]
        self._step = max(0, min(start_index, len(self._seq)))
        self._bind_name(self.current_name)

    def advance(self):
        if not self._seq:
            return
        self._step += 1
        if self._step >= len(self._seq):
            self._step = len(self._seq)
            self._unbind_all()
        else:
            self._bind_name(self.current_name)

    def refresh_after_reset(self):
        self._step = 0
        self._bind_name(self.current_name)

    @property
    def current_name(self) -> Optional[str]:
        if 0 <= self._step < len(self._seq):
            return self._seq[self._step]
        return None

    @property
    def step_index(self) -> int:
        return self._step

    @property
    def total_steps(self) -> int:
        return len(self._seq)

# ---------------------------------------------------------------------------
# Physics binder: bind physics material to all names in SEQUENCE at once
# ---------------------------------------------------------------------------

class PhysicsSequenceBinder:
    """
    Binds a physics material once per asset root (by leaf name) across all envs.
    Call bind_now() after spawn; call refresh_after_reset() after env reset.
    """
    def __init__(self, sequence: List[str], phys_mat_path: str):
        self._seq = sequence[:] if sequence else []
        self._mat_path = phys_mat_path
        self._last_targets: list[str] = []

    def bind_now(self):
        # Resolve all asset roots for each name in SEQUENCE; bind at the root
        targets: list[str] = []
        for name in self._seq:
            targets.extend(_find_asset_roots_by_leaf_name(name))
        # de-dupe while preserving order
        seen, uniq = set(), []
        for p in targets:
            if p not in seen:
                uniq.append(p); seen.add(p)
        for p in uniq:
            sim_utils.make_uninstanceable(p)
            sim_utils.bind_physics_material(p, self._mat_path)
        self._last_targets = uniq

    def refresh_after_reset(self):
        self.bind_now()

# ----------------------- helpers -----------------------

def _find_asset_roots_by_leaf_name(leaf_name: str) -> list[str]:
    """
    Returns asset root prim paths whose leaf == leaf_name across all envs.
    Uses regex match, so no need to traverse children.
    """
    pattern = f"/World/envs/env_.*/{leaf_name}$"
    return sim_utils.find_matching_prim_paths(pattern)

def _find_visuals_path(stage: Usd.Stage, asset_root_path: str) -> Optional[str]:
    """
    Prefer a direct child named 'visuals' under the asset root.
    Fallback: any descendant named 'visuals'. Returns the prim path or None.
    """
    root = stage.GetPrimAtPath(asset_root_path)
    if not root or not root.IsValid():
        return None

    # 1) direct child named 'visuals'
    for child in root.GetChildren():
        if child.GetName().lower() == "visuals" and child.IsValid():
            return str(child.GetPath())

    # 2) fallback: any descendant named 'visuals'
    for p in Usd.PrimRange(root):
        if p != root and p.GetName().lower() == "visuals":
            return str(p.GetPath())

    return None

def ang_deg(q1: Gf.Quatd, q2: Gf.Quatd) -> float:
    dq = q1 * q2.GetInverse()
    a = 2.0 * math.degrees(math.acos(max(-1.0, min(1.0, abs(dq.GetReal())))))
    return min(a, 360.0 - a)

def first_descendant_with_rigid_body(stage: Usd.Stage, root_prim: Usd.Prim) -> Optional[Usd.Prim]:
    if not root_prim or not root_prim.IsValid():
        return None
    if UsdPhysics.RigidBodyAPI(root_prim):
        return root_prim
    for p in Usd.PrimRange(root_prim):
        if UsdPhysics.RigidBodyAPI(p):
            return p
    return None

def resolve_env_scoped_path(stage: Usd.Stage, env_root_path: str, leaf_name: str) -> Optional[str]:
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

def physx_get_pose(prim_path: str) -> Optional[Tuple[Gf.Vec3d, Gf.Quatd]]:
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

def get_xform_scale(stage: Usd.Stage, prim_path: str) -> Gf.Vec3d:
    """
    Read a local scale op from the prim if present, otherwise return (1,1,1).
    Assumes uniform scaling via UsdGeom.XformOp(TypeScale).
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return Gf.Vec3d(1.0, 1.0, 1.0)
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            s = op.Get()
            return Gf.Vec3d(float(s[0]), float(s[1]), float(s[2]))
    return Gf.Vec3d(1.0, 1.0, 1.0)

def spawn_ghost_preview(
    stage: Usd.Stage,
    source_root_path: str,
    target_pos: Gf.Vec3d,
    target_rot: Gf.Quatd,
    ghost_root_path: str,
    ghost_mat_path: str,
) -> str:
    """
    Create/update a visual-only ghost of `source_root_path` at `target_pos,target_rot`.

    - It references the source's 'visuals' child (if any), otherwise the source root.
    - It binds a ghost material at the ghost root.
    - Returns the ghost root prim path.
    """
    # Ensure ghost root exists as an Xform
    ghost_xf = UsdGeom.Xform.Define(stage, ghost_root_path)
    ghost_prim = ghost_xf.GetPrim()

    # Reference only visuals subtree if possible
    visuals_path = _find_visuals_path(stage, source_root_path) or source_root_path
    ghost_prim.GetReferences().ClearReferences()
    ghost_prim.GetReferences().AddReference(
        stage.GetRootLayer().identifier,
        visuals_path,
    )
    
    # Build transform with scale -> rot -> translate
    source_scale = get_xform_scale(stage, source_root_path)
    scale = source_scale if source_scale is not None else Gf.Vec3d(1.0, 1.0, 1.0)

    xformable = UsdGeom.Xformable(ghost_prim)
    xformable.ClearXformOpOrder()
    op = xformable.AddTransformOp()

    # Apply scale
    sM = Gf.Matrix4d(1.0)
    sM.SetScale(scale)
    # Apply rotation
    rotM = Gf.Matrix4d(1.0)
    rotM.SetRotate(target_rot)
    # Apply translation
    tM = Gf.Matrix4d(1.0)
    tM.SetTranslate(target_pos)
    
    m = sM * rotM * tM

    op.Set(m)

    # Bind ghost material
    sim_utils.bind_visual_material(ghost_root_path, ghost_mat_path, stronger_than_descendants=True)
    return ghost_root_path

# ---------------------------------------------------------------------------
# Minimal HUD
# ---------------------------------------------------------------------------

import omni.ui as ui
import omni.ui.scene as sc

# class SimpleSceneWidget(ui.Widget):
#     def __init__(self, text="Hello", **kwargs):
#         super().__init__(**kwargs)
#         with ui.ZStack():
#             ui.Rectangle(style={
#                 "background_color": ui.color("#292929"),
#                 "border_color": ui.color(0.7),
#                 "border_width": 1,
#                 "border_radius": 2,
#             })
#             with ui.VStack(style={"margin": 5}):
#                 self.label = ui.Label(text, alignment=ui.Alignment.CENTER, style={"font_size": 4})
                
class SimpleSceneWidget(ui.Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._root = None

        with ui.ZStack():
            ui.Rectangle(style={
                "background_color": ui.color("#292929"),
                "border_color": ui.color(0.7),
                "border_width": 0.5,
                "border_radius": 1,
            })
            # vertical list of steps
            with ui.VStack(height=1, style={"margin": 1, "spacing": 1}) as root:
                self._root = root

    def set_steps(self, wrapped_steps: list[str], active_index: int):
        """
        wrapped_steps: list of strings that may already contain '\\n' for wrapping.
        active_index: index of current step.
        """
        if self._root is None:
            return

        self._root.clear()

        with self._root:
            for i, text in enumerate(wrapped_steps):
                is_active = (i == active_index)

                style = {
                    "font_size": 1,
                    "color": ui.color("#83ff6d") if is_active else ui.color("#f5f5f5"),
                    "margin": 1,
                    "margin_height": 0,
                    "margin_width": 0,
                    "padding": 1,
                }
                if is_active:
                    style["border_radius"] = 2

                ui.Label(text, height=3, style=style)

from omni.kit.xr.scene_view.utils.ui_container import UiContainer
from omni.kit.xr.scene_view.utils.manipulator_components.widget_component import WidgetComponent
from omni.kit.xr.scene_view.utils.spatial_source import SpatialSource

# class HUDManager:
#     def __init__(
#         self,
#         widget_cls,
#         width: float = 4.0,
#         height: float = 1.3,
#         resolution_scale: int = 10,
#         unit_to_pixel_scale: int = 20,
#         translation: Gf.Vec3d = Gf.Vec3d(0, 1.5, 2),
#         rotation_deg_xyz: Gf.Vec3d = Gf.Vec3d(90, 0, 0),
#     ):
#         self._widget = None
#         self._label = None

#         def on_constructed(widget_instance):
#             self._widget = widget_instance
#             self._label = getattr(widget_instance, "label", None)

#         self._widget_component = WidgetComponent(
#             widget_cls,
#             width=width,
#             height=height,
#             resolution_scale=resolution_scale,
#             unit_to_pixel_scale=unit_to_pixel_scale,
#             update_policy=sc.Widget.UpdatePolicy.ALWAYS,
#             construct_callback=on_constructed,
#         )
#         space_stack = [
#             SpatialSource.new_translation_source(translation),
#             SpatialSource.new_rotation_source(Gf.Vec3d(
#                 math.radians(rotation_deg_xyz[0]),
#                 math.radians(rotation_deg_xyz[1]),
#                 math.radians(rotation_deg_xyz[2]),
#             )),
#         ]
#         self._ui_container = UiContainer(self._widget_component, space_stack=space_stack)

#     def get_widget_dimensions(self, text: str, font_size: float, max_width: float, min_width: float):
#         # Estimate average character width.
#         char_width = 0.03 * font_size
#         max_chars_per_line = int(max_width / char_width)
#         lines = textwrap.wrap(text, width=max_chars_per_line)
#         if not lines:
#             lines = [text]
#         wrapped_text = "\n".join(lines)
#         return wrapped_text
    
#     def update(self, text: str):
#         if self._label:
#             self._label.text = self.get_widget_dimensions(text, 4.0, 4.0, 4.0)

#     def show(self):  self._widget_component.visible = True
#     def hide(self):  self._widget_component.visible = False
#     def destroy(self): pass

class HUDManager:
    def __init__(
        self,
        widget_cls,
        width: float = 0.6,   # narrower
        height: float = 0.6,  # taller
        resolution_scale: int = 20,
        unit_to_pixel_scale: int = 30,
        translation: Gf.Vec3d = Gf.Vec3d(0, 0.9, 1.5),
        rotation_deg_xyz: Gf.Vec3d = Gf.Vec3d(90, 0, 0),
    ):
        self._widget = None
        self._width = width
        self._font_size = 1.5 

        def on_constructed(widget_instance):
            self._widget = widget_instance

        self._widget_component = WidgetComponent(
            widget_cls,
            width=width,
            height=height,
            resolution_scale=resolution_scale,
            unit_to_pixel_scale=unit_to_pixel_scale,
            update_policy=sc.Widget.UpdatePolicy.ALWAYS,
            construct_callback=on_constructed,
        )
        space_stack = [
            SpatialSource.new_translation_source(translation),
            SpatialSource.new_rotation_source(Gf.Vec3d(
                math.radians(rotation_deg_xyz[0]),
                math.radians(rotation_deg_xyz[1]),
                math.radians(rotation_deg_xyz[2]),
            )),
        ]
        self._ui_container = UiContainer(self._widget_component, space_stack=space_stack)

    def wrap_text(self, text: str) -> str:
        char_width = 0.009 * self._font_size
        usable_width = self._width * 0.9
        max_chars_per_line = max(10, int(usable_width / char_width))
        lines = textwrap.wrap(text, width=max_chars_per_line)
        return "\n".join(lines) if lines else text
    
    def update(self, guide: "BaseGuide", highlighter: VisualSequenceHighlighter):
        if not self._widget or not hasattr(guide, "get_all_instructions"):
            return

        steps = guide.get_all_instructions()
        if not steps:
            return
        
        total_real = len(guide.SEQUENCE)          # 4
        idx = highlighter.step_index   

        wrapped_lines: list[str] = []

        if idx >= total_real:
            # Completed: mark all real steps as done, final line active
            for i, s in enumerate(steps):
                if i < total_real:
                    full = f"[x]  {s}"
                else:
                    full = s  # "Assembly complete!"
                wrapped_lines.append(self.wrap_text(full))
            active_idx = len(steps) - 1  # highlight "Assembly complete!"
        else:
            # In progress
            for i, s in enumerate(steps):
                if i < total_real:
                    done = i < idx
                    marker = "[x] " if done else "[ ] "
                    full = f"{marker} {s}"
                else:
                    # final line shown but not done yet
                    full = s
                wrapped_lines.append(self.wrap_text(full))
            active_idx = idx  # highlight current real step

        if hasattr(self._widget, "set_steps"):
            self._widget.set_steps(wrapped_lines, active_idx)

    def show(self):  self._widget_component.visible = True
    def hide(self):  self._widget_component.visible = False
    def destroy(self): pass

# ---------------------------------------------------------------------------
# Base guide
# ---------------------------------------------------------------------------

class BaseGuide:
    SEQUENCE: List[str] = []  # override in subclasses
    
    def __init__(self):
        self._stage: Optional[Usd.Stage] = None
        # Guides can fill this after on_reset:
        # logical_name (e.g. "DrawerBottom") -> ghost prim path
        self._ghost_paths_by_name: dict[str, str] = {}
    
    def on_reset(self, env):
        self._stage = env.scene.stage
    
    def get_all_instructions(self) -> List[str]:
        """
        Default instruction list derived from SEQUENCE.
        Subclasses can override to provide nicer text.
        """
        total = len(self.SEQUENCE) or 1
        return [
            f"Step {i+1}/{total}: {name}"
            for i, name in enumerate(self.SEQUENCE)
        ]

    # def step_label(self, highlighter: VisualSequenceHighlighter) -> str:
    #     """
    #     Backwards-compatible single-line label using the instruction list.
    #     """
    #     labels = self.get_all_instructions()
    #     idx = highlighter.step_index
    #     if 0 <= idx < len(labels):
    #         return labels[idx]
    #     return "Assembly complete!"
    
    def get_live_part_pose(self, name: str):
        p = getattr(self, "_paths", {}).get(name)
        if not p:
            return None
        return physx_get_pose(p)

    def create_highlighter(self, stage: Usd.Stage) -> VisualSequenceHighlighter:
        MaterialRegistry.ensure_all(stage)  # make sure materials exist
        hl = VisualSequenceHighlighter(stage, self.SEQUENCE, MaterialRegistry.visual_path)
        hl.set_sequence(self.SEQUENCE, start_index=0)
        return hl

    def create_physics_binder(self) -> PhysicsSequenceBinder:
        return PhysicsSequenceBinder(self.SEQUENCE, MaterialRegistry.physics_path)

    def maybe_auto_advance(self, highlighter: VisualSequenceHighlighter):
        """
        Call once per sim tick *after* env.step() or sim.render().
        Uses PhysX for moving parts, cached USD for statics.
        """
        idx = highlighter.step_index
        checks: Sequence[Callable[[], bool]] | None = getattr(self, "_checks", None)
        if not checks:
            return

        if idx >= len(checks):
            return
        
        if checks[idx]():
            highlighter.advance()
            
    def _ghost_name_for_step(self, step_index: int) -> Optional[str]:
        """
        Default mapping from step index to the logical name whose ghost should be visible.
        Subclasses can override if they want different behaviour.
        """
        if 0 <= step_index < len(self.SEQUENCE):
            return self.SEQUENCE[step_index]
        return None
    
    def _update_ghost_visibility_for_step(self, step_index: int) -> None:
        """
        Show only the ghost corresponding to this step; hide others.
        Uses self._ghost_paths_by_name which subclasses should fill in on_reset.
        """
        if self._stage is None:
            return

        current_name = self._ghost_name_for_step(step_index)

        for name, ghost_path in self._ghost_paths_by_name.items():
            prim = self._stage.GetPrimAtPath(ghost_path)
            if not prim or not prim.IsValid():
                continue
            img = UsdGeom.Imageable(prim)
            if name == current_name:
                img.MakeVisible()
            else:
                img.MakeInvisible()

    def update_previews_for_step(self, highlighter: VisualSequenceHighlighter) -> None:
        """
        Called from the teleop loop; updates which ghost is visible.
        Subclasses normally just populate self._ghost_paths_by_name.
        """
        self._update_ghost_visibility_for_step(highlighter.step_index)

