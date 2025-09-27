from __future__ import annotations
from typing import List, Optional
import math
from pxr import Usd, UsdShade, Sdf, Gf
import omni.ui as ui
import omni.kit.app
import omni.replicator.core as rep
import omni.ui.scene as sc
from omni.kit.xr.scene_view.utils.ui_container import UiContainer
from omni.kit.xr.scene_view.utils.manipulator_components.widget_component import WidgetComponent
from omni.kit.xr.scene_view.utils.spatial_source import SpatialSource

# ---------- Material utils ----------

def find_prim_paths_by_name(stage: Usd.Stage, name: str) -> list[str]:
    """Return all prim paths whose leaf name == name (robust across USD versions)."""
    hits = []
    for prim in Usd.PrimRange.Stage(stage):
        # Keep it simple & version-proof; avoid GeomSubset/other type checks
        if not prim.IsValid() or not prim.IsActive():
            continue
        if prim.GetName() == name:
            hits.append(prim.GetPath().pathString)
    return hits

def create_preview_surface_material(stage: Usd.Stage, prim_path: str,
                                    diffuse=(0.1, 0.7, 0.1),
                                    emissive=(8.0, 8.0, 1.0),
                                    roughness=0.6,
                                    metallic=0.0) -> UsdShade.Material:
    """Create/get a simple UsdPreviewSurface material at prim_path and bindable surface output."""
    # Define material + shader prims
    mat_prim = stage.DefinePrim(prim_path, "Material")
    material = UsdShade.Material(mat_prim)

    shader_prim = stage.DefinePrim(f"{prim_path}/Shader", "Shader")
    shader = UsdShade.Shader(shader_prim)
    shader.CreateIdAttr("UsdPreviewSurface")

    # Inputs
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    shader.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(float(metallic))

    # Create the shader's *output* explicitly and connect output→output
    shader_surface_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader_surface_out)

    return material

class MaterialSwap:
    """Save/restore original material bindings and bind a provided USD material on demand."""
    def __init__(self, stage: Usd.Stage):
        self._stage = stage
        self._saved: dict[str, Optional[UsdShade.Material]] = {}

    def _get_bound_material(self, prim: Usd.Prim):
        db = UsdShade.MaterialBindingAPI(prim).GetDirectBinding()
        return db.GetMaterial()

    def apply(self, prim_path: str, material: UsdShade.Material, stronger_than_descendants: bool = True):
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return
        if prim_path not in self._saved:
            self._saved[prim_path] = self._get_bound_material(prim)
        api = UsdShade.MaterialBindingAPI(prim)
        strength = UsdShade.Tokens.strongerThanDescendants if stronger_than_descendants else UsdShade.Tokens.weakerThanDescendants
        api.Bind(material, bindingStrength=strength)

    def restore(self, prim_path: str):
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return
        prev = self._saved.get(prim_path, None)
        api = UsdShade.MaterialBindingAPI(prim)
        if prev:
            api.Bind(prev)
        else:
            api.UnbindDirectBinding()
        if prim_path in self._saved:
            del self._saved[prim_path]

class MaterialHighlighter:
    """Highlights a sequence of part names; supports step advance + refresh on env reset."""
    def __init__(self, stage: Usd.Stage, material: UsdShade.Material):
        self.stage = stage
        self.material = material
        self._swap = MaterialSwap(stage)
        self._sequence: List[str] = []
        self._step: int = 0
        self._current_paths: List[str] = []
        self._current_name: Optional[str] = None

    def set_sequence(self, names: List[str], start_index: int = 0):
        self.clear()
        self._sequence = names[:]
        self._step = max(0, min(start_index, len(self._sequence)))
        if self._sequence:
            self.highlight_name(self._sequence[self._step])

    def highlight_name(self, name: Optional[str]):
        self.clear()
        self._current_name = name
        if not name:
            return
        self._current_paths = find_prim_paths_by_name(self.stage, name)
        for p in self._current_paths:
            self._swap.apply(p, self.material, stronger_than_descendants=True)

    def clear(self):
        for p in self._current_paths:
            self._swap.restore(p)
        self._current_paths = []

    def advance(self):
        if not self._sequence:
            return
        self._step += 1
        if self._step >= len(self._sequence):
            self.clear()
            self._current_name = None
            self._step = len(self._sequence)
            return
        self.highlight_name(self._sequence[self._step])

    def refresh_current(self):
        name = self._current_name
        if name:
            self.highlight_name(name)

    @property
    def current_name(self) -> Optional[str]:
        return self._current_name

    @property
    def step_index(self) -> int:
        return self._step

    @property
    def total_steps(self) -> int:
        return len(self._sequence)

# ---------- Minimal HUD ----------

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
                self.label = ui.Label(text, alignment=ui.Alignment.CENTER, style={"font_size": 5})

class HUDManager:
    """Owns the WidgetComponent + UiContainer + spatial sources.
    Exposes update(text), show(), hide(), and destroy()."""

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
        self._ui_container = None

        def on_constructed(widget_instance):
            self._widget = widget_instance
            # Expect SimpleSceneWidget from base.py that exposes .label
            self._label = getattr(widget_instance, "label", None)

        self._widget_component = WidgetComponent(
            widget_cls,
            width=width,
            height=height,
            resolution_scale=resolution_scale,
            unit_to_pixel_scale=unit_to_pixel_scale,
            update_policy=sc.Widget.UpdatePolicy.ON_DEMAND,
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

    def update(self, text: str):
        if self._label:
            self._label.text = text

    def show(self):
        if self._ui_container:
            self._ui_container.visible = True

    def hide(self):
        if self._ui_container:
            self._ui_container.visible = False

    def destroy(self):
        # If your UiContainer/WidgetComponent require explicit cleanup, do it here.
        self._ui_container = None
        self._widget = None
        self._label = None

# ---------- Base guide ----------

class BaseGuide:
    """
    A guide owns:
      - the step sequence of part names to highlight
      - how to render/update the HUD text for each step
      - optional auto-advance rules (override if desired)
    """
    # Override in subclasses
    SEQUENCE: List[str] = []

    def make_highlight_material(self, stage: Usd.Stage) -> UsdShade.Material:
        return create_preview_surface_material(stage, "/World/Materials/Highlight", diffuse=(0.6, 0.8, 0.1), emissive=(0, 0, 0))

    def create_highlighter(self, stage: Usd.Stage) -> MaterialHighlighter:
        mat = self.make_highlight_material(stage)
        highlighter = MaterialHighlighter(stage, mat)
        highlighter.set_sequence(self.SEQUENCE, start_index=0)
        return highlighter

    def step_label(self, highlighter: MaterialHighlighter) -> str:
        idx = highlighter.step_index
        total = highlighter.total_steps or 1
        name = highlighter.current_name or "Done"
        return f"Step {min(idx+1, total)}/{total}: Pick up {name}"

    # Optionally override for automatic step changes
    def maybe_auto_advance(self, env, highlighter: MaterialHighlighter):
        """Called every frame; default: no-op."""
        return
