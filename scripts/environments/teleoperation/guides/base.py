# base.py
from __future__ import annotations
import textwrap
from typing import List, Optional

from pxr import Usd, Gf
from isaaclab.sim import utils as sim_utils
from isaaclab.sim.spawners.materials import spawn_preview_surface, spawn_rigid_body_material
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg

from pxr import UsdShade

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

    @classmethod
    def ensure_all(cls, stage: Usd.Stage) -> None:
        # Create/update visual material
        spawn_preview_surface(prim_path=cls.visual_path, cfg=cls.visual_cfg)
        # Create/update physics material
        spawn_rigid_body_material(prim_path=cls.physics_path, cfg=cls.physics_cfg)

# ---------------------------------------------------------------------------
# Resolve prim roots by leaf name across all env instances
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Unbinds visual materials
# ---------------------------------------------------------------------------

# def _unbind_visual_material(stage: Usd.Stage, prim_path: str):
#     prim = stage.GetPrimAtPath(prim_path)
#     if not prim or not prim.IsValid():
#         return
#     UsdShade.MaterialBindingAPI(prim).UnbindDirectBinding()

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

        # map each root to its 'visuals' path; if missing, fall back to root (last resort)
        targets = []
        for root_path in asset_roots:
            vpath = _find_visuals_path(self._stage, root_path)
            targets.append(vpath if vpath else root_path)

        # de-dupe while preserving order
        seen, uniq = set(), []
        for p in targets:
            if p not in seen:
                uniq.append(p); seen.add(p)

        # bind only on visuals (or root if we had to fall back)
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

# ---------------------------------------------------------------------------
# Minimal HUD (unchanged)
# ---------------------------------------------------------------------------

import omni.ui as ui
import omni.ui.scene as sc
import math

class SimpleSceneWidget(ui.Widget):
    def __init__(self, text="Hello", **kwargs):
        super().__init__(**kwargs)
        with ui.ZStack():
            ui.Rectangle(style={
                "background_color": ui.color("#292929"),
                "border_color": ui.color(0.7),
                "border_width": 1,
                "border_radius": 2,
            })
            with ui.VStack(style={"margin": 5}):
                self.label = ui.Label(text, alignment=ui.Alignment.CENTER, style={"font_size": 4})

from omni.kit.xr.scene_view.utils.ui_container import UiContainer
from omni.kit.xr.scene_view.utils.manipulator_components.widget_component import WidgetComponent
from omni.kit.xr.scene_view.utils.spatial_source import SpatialSource

class HUDManager:
    def __init__(
        self,
        widget_cls,
        width: float = 4.0,
        height: float = 1.3,
        resolution_scale: int = 10,
        unit_to_pixel_scale: int = 20,
        translation: Gf.Vec3d = Gf.Vec3d(0, 1.5, 2),
        rotation_deg_xyz: Gf.Vec3d = Gf.Vec3d(90, 0, 0),
    ):
        self._widget = None
        self._label = None

        def on_constructed(widget_instance):
            self._widget = widget_instance
            self._label = getattr(widget_instance, "label", None)

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

    def get_widget_dimensions(self, text: str, font_size: float, max_width: float, min_width: float):
        #Estimate average character width.
        char_width = 0.03 * font_size
        max_chars_per_line = int(max_width / char_width)
        lines = textwrap.wrap(text, width=max_chars_per_line)
        if not lines:
            lines = [text]
        # computed_width = max(len(line) for line in lines) * char_width
        # actual_width = max(min(computed_width, max_width), min_width)
        # line_height = 1.2 * font_size
        # actual_height = len(lines) * line_height
        wrapped_text = "\n".join(lines)
        return wrapped_text
    
    def update(self, text: str):
        if self._label:
            self._label.text = self.get_widget_dimensions(text, 4.0, 4.0, 4.0)

    def show(self):  self._widget_component.visible = True
    def hide(self):  self._widget_component.visible = False
    def destroy(self): pass

# ---------------------------------------------------------------------------
# Base guide
# ---------------------------------------------------------------------------

class BaseGuide:
    SEQUENCE: List[str] = []  # override in subclasses

    def create_highlighter(self, stage: Usd.Stage) -> VisualSequenceHighlighter:
        MaterialRegistry.ensure_all(stage)  # make sure materials exist
        hl = VisualSequenceHighlighter(stage, self.SEQUENCE, MaterialRegistry.visual_path)
        hl.set_sequence(self.SEQUENCE, start_index=0)
        return hl

    def create_physics_binder(self) -> PhysicsSequenceBinder:
        return PhysicsSequenceBinder(self.SEQUENCE, MaterialRegistry.physics_path)

    # def step_label(self, highlighter: VisualSequenceHighlighter) -> str:
    #     idx = highlighter.step_index
    #     total = highlighter.total_steps or 1
    #     name = highlighter.current_name or "Done"
    #     return f"Step {min(idx+1, total)}/{total}: Pick up {name}"

    def maybe_auto_advance(self, env, highlighter: VisualSequenceHighlighter):
        return
