from __future__ import annotations
import textwrap
from typing import Callable, List, Optional, Sequence, Tuple
from pxr import Usd, Gf, UsdPhysics, UsdGeom
from isaaclab.sim import utils as sim_utils
from isaaclab.sim.spawners.materials import (
    spawn_preview_surface,
    spawn_rigid_body_material,
    spawn_from_mdl_file,
)
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import (
    PreviewSurfaceCfg,
    GlassMdlCfg,
)
from pxr import UsdShade
from omni.physx import get_physx_interface
import math
import carb

# Material registry


class MaterialRegistry:

    visual_path = "/World/Materials/Highlight"
    physics_path = "/World/Materials/Grasp"
    ghost_path = "/World/Materials/GhostPreview"

    # Visual highlighter material
    visual_cfg = PreviewSurfaceCfg(
        diffuse_color=(0.6, 0.8, 0.1),
    )

    # Physics material
    physics_cfg = RigidBodyMaterialCfg(
        static_friction=0.8,
        dynamic_friction=0.7,
        friction_combine_mode="multiply",
    )

    # Ghost preview material
    # ghost_cfg = GlassMdlCfg(
    #     glass_color=(0.871, 1, 0.957),
    #     frosting_roughness=0,
    #     thin_walled=True,
    #     glass_ior=1.8,
    # )

    ghost_cfg = PreviewSurfaceCfg(
        diffuse_color=(0.4, 1.0, 1.0),
        roughness=1.0,
    )

    @classmethod
    def ensure_all(cls, stage: Usd.Stage) -> None:
        # Create/update visual material
        spawn_preview_surface(prim_path=cls.visual_path, cfg=cls.visual_cfg)
        # Create/update physics material
        spawn_rigid_body_material(prim_path=cls.physics_path, cfg=cls.physics_cfg)
        # Create/update host preview material
        # spawn_from_mdl_file(prim_path=cls.ghost_path, cfg=cls.ghost_cfg)
        spawn_preview_surface(prim_path=cls.ghost_path, cfg=cls.ghost_cfg)


# Visual highlighter


class VisualSequenceHighlighter:

    def __init__(self, stage: Usd.Stage, sequence: List[str], visual_mat_path: str):
        self._stage = stage
        self._seq = sequence[:] if sequence else []
        self._mat_path = visual_mat_path
        self._step = 0
        self._active_visuals_paths: list[str] = []

    def _unbind_all(self):
        for p in self._active_visuals_paths:
            # only unbind where bound before
            UsdShade.MaterialBindingAPI(
                self._stage.GetPrimAtPath(p)
            ).UnbindDirectBinding()
        self._active_visuals_paths = []

    def _bind_name(self, leaf_name: Optional[str]):
        self._unbind_all()
        if not leaf_name:
            return

        # resolve all asset roots for leaf across envs
        asset_roots = find_asset_roots_by_leaf_name(leaf_name)

        # map each root to its visuals path; if missing fall back to root
        targets = []
        for root_path in asset_roots:
            vpath = find_visuals_path(self._stage, root_path)
            targets.append(vpath if vpath else root_path)

        seen, uniq = set(), []
        for p in targets:
            if p not in seen:
                uniq.append(p)
                seen.add(p)

        # bind only on visuals
        for p in uniq:
            sim_utils.bind_visual_material(
                p, self._mat_path, stronger_than_descendants=True
            )

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


# Physics binder


class PhysicsSequenceBinder:

    def __init__(self, sequence: List[str], phys_mat_path: str):
        self._seq = sequence[:] if sequence else []
        self._mat_path = phys_mat_path
        self._last_targets: list[str] = []

    def bind_now(self):
        # Resolve all asset roots for each name in SEQUENCE; bind at the root
        targets: list[str] = []
        for name in self._seq:
            targets.extend(find_asset_roots_by_leaf_name(name))
        seen, uniq = set(), []
        for p in targets:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        for p in uniq:
            sim_utils.make_uninstanceable(p)
            sim_utils.bind_physics_material(p, self._mat_path)
        self._last_targets = uniq

    def refresh_after_reset(self):
        self.bind_now()


class StepHighlighter(Protocol):
    @property
    def step_index(self) -> int: ...

    @property
    def total_steps(self) -> int: ...

    def advance(self) -> None: ...

    def refresh_after_reset(self) -> None: ...


# ----------------------- helpers -----------------------


def find_asset_roots_by_leaf_name(leaf_name: str) -> list[str]:

    pattern = f"/World/envs/env_.*/{leaf_name}$"
    return sim_utils.find_matching_prim_paths(pattern)


def find_visuals_path(stage: Usd.Stage, asset_root_path: str) -> Optional[str]:

    root = stage.GetPrimAtPath(asset_root_path)
    if not root or not root.IsValid():
        return None

    # direct child named visuals
    for child in root.GetChildren():
        if child.GetName().lower() == "visuals" and child.IsValid():
            return str(child.GetPath())

    # fallback to any descendant named visuals
    for p in Usd.PrimRange(root):
        if p != root and p.GetName().lower() == "visuals":
            return str(p.GetPath())

    return None


def ang_deg(q1: Gf.Quatd, q2: Gf.Quatd) -> float:
    dq = q1 * q2.GetInverse()
    a = 2.0 * math.degrees(math.acos(max(-1.0, min(1.0, abs(dq.GetReal())))))
    return min(a, 360.0 - a)


def first_descendant_with_rigid_body(
    stage: Usd.Stage, root_prim: Usd.Prim
) -> Optional[Usd.Prim]:
    if not root_prim or not root_prim.IsValid():
        return None
    if UsdPhysics.RigidBodyAPI(root_prim):
        return root_prim
    for p in Usd.PrimRange(root_prim):
        if UsdPhysics.RigidBodyAPI(p):
            return p
    return None


def resolve_env_scoped_path(
    stage: Usd.Stage, env_root_path: str, leaf_name: str
) -> Optional[str]:
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

    # Ensure ghost root exists as an Xform
    ghost_xf = UsdGeom.Xform.Define(stage, ghost_root_path)
    ghost_prim = ghost_xf.GetPrim()

    # Reference only visuals subtree
    visuals_path = find_visuals_path(stage, source_root_path) or source_root_path
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
    sim_utils.bind_visual_material(
        ghost_root_path, ghost_mat_path, stronger_than_descendants=True
    )
    return ghost_root_path


def isaac_world_to_xr_ui(pos_xyz):  # pos_xyz is Gf.Vec3d or tuple/list (x,y,z)
    x = float(pos_xyz[0])  # left/right
    y = float(pos_xyz[1])  # forward (Isaac)
    z = float(pos_xyz[2])  # up (Isaac)

    # Common mapping if XR expects: X right, Y up, -Z forward
    return carb.Float3(x, z, -y)


# Minimal HUD

import omni.ui as ui
import omni.ui.scene as sc


class SimpleSceneWidget(ui.Widget):
    def __init__(self, max_lines: int = 10, **kwargs):
        super().__init__(**kwargs)
        self._labels: list[ui.Label] = []
        self._max_lines = max_lines

        with ui.ZStack():
            ui.Rectangle(
                style={
                    "background_color": ui.color("#292929"),
                    "border_color": ui.color(0.7),
                    "border_width": 0.5,
                    "border_radius": 1,
                }
            )
            with ui.VStack(height=1, style={"margin": 1, "spacing": 1}):
                for _ in range(max_lines):
                    lbl = ui.Label(
                        "",
                        word_wrap=True,
                        height=3,
                        style={"font_size": 1, "color": ui.color("#f5f5f5")},
                    )
                    lbl.visible = False
                    self._labels.append(lbl)

    def set_steps(self, wrapped_steps: list[str], active_index: int):
        # Ensure we have enough labels (optional: grow if needed)
        if len(wrapped_steps) > self._max_lines:
            # if you expect more lines, increase max_lines at construction time
            wrapped_steps = wrapped_steps[: self._max_lines]

        for i, lbl in enumerate(self._labels):
            if i < len(wrapped_steps):
                lbl.visible = True
                lbl.text = wrapped_steps[i]
                is_active = i == active_index
                lbl.style = {
                    "font_size": 1,
                    "color": ui.color("#83ff6d") if is_active else ui.color("#f5f5f5"),
                    "margin": 1,
                    "margin_height": 0,
                    "margin_width": 0,
                    "padding": 1,
                }
            else:
                lbl.visible = False


from omni.kit.xr.scene_view.utils.ui_container import UiContainer
from omni.kit.xr.scene_view.utils.manipulator_components.widget_component import (
    WidgetComponent,
)
from omni.kit.xr.scene_view.utils.spatial_source import SpatialSource


class HUDManager:
    def __init__(
        self,
        widget_cls,
        width: float = 0.6,
        height: float = 0.8,
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
            SpatialSource.new_rotation_source(
                Gf.Vec3d(
                    math.radians(rotation_deg_xyz[0]),
                    math.radians(rotation_deg_xyz[1]),
                    math.radians(rotation_deg_xyz[2]),
                )
            ),
        ]
        self._ui_container = UiContainer(
            self._widget_component, space_stack=space_stack
        )

    def wrap_text(self, text: str) -> str:
        char_width = 0.009 * self._font_size
        usable_width = self._width * 0.9
        max_chars_per_line = max(10, int(usable_width / char_width))
        wrapped_lines: list[str] = []
        for line in text.splitlines() or [""]:
            chunks = textwrap.wrap(line, width=max_chars_per_line) or [""]
            wrapped_lines.extend(chunks)

        return "\n".join(wrapped_lines)

    def update(self, guide: "BaseGuide", highlighter: StepHighlighter):
        if not self._widget or not hasattr(guide, "get_all_instructions"):
            return

        steps = guide.get_all_instructions()
        if not steps:
            return

        total_real = len(guide.SEQUENCE)
        idx = highlighter.step_index

        wrapped_lines: list[str] = []

        if idx >= total_real:
            # Verification phase
            for i, s in enumerate(steps):
                if i < total_real:
                    full = f"[x]  {s}"
                else:
                    full = s
                wrapped_lines.append(self.wrap_text(full))

            # Final line based on global validity
            global_ok = True
            if hasattr(guide, "is_final_assembly_valid"):
                try:
                    global_ok = bool(guide.is_final_assembly_valid())
                except Exception:
                    global_ok = True

            if global_ok:
                final_text = "Assembly complete!"
            else:
                issues: list[tuple[str, str]] = []
                if hasattr(guide, "final_unmet_constraints"):
                    try:
                        issues = list(guide.final_unmet_constraints())
                    except Exception:
                        issues = []

                # show only first 1–2 messages to keep XR readable
                msgs = [msg for _, msg in issues]
                if msgs:
                    short = "\n".join(f"- {m}" for m in msgs[:2])
                    final_text = f"Fix:\n{short}"

                else:
                    final_text = "Fix:\n- Alignment check failed. Please adjust parts"

            wrapped_lines[-1] = self.wrap_text(final_text)
            active_idx = len(steps) - 1
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
            active_idx = idx  # highlight current step

        if hasattr(self._widget, "set_steps"):
            self._widget.set_steps(wrapped_lines, active_idx)

    def show(self):
        self._widget_component.visible = True

    def hide(self):
        self._widget_component.visible = False

    def destroy(self):
        try:
            self.hide()
        except Exception:
            pass
        self._widget = None
        try:
            if hasattr(self._ui_container, "destroy"):
                self._ui_container.destroy()
        except Exception:
            pass
        self._ui_container = None
        self._widget_component = None


# NameTag Widget


class NameTagWidget(ui.Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._label = None

        with ui.ZStack():
            # Border + background
            ui.Rectangle(
                style={
                    "background_color": ui.color("#292929"),
                    "border_color": ui.color("#ffffff"),
                    "border_width": 0.1,
                    "border_radius": 0.2,
                }
            )

            # Content container (use VStack)
            with ui.VStack(height=1, style={"margin": 0.01, "spacing": 0.01}):
                self._label = ui.Label(
                    "",
                    word_wrap=False,
                    alignment=ui.Alignment.CENTER,
                    style={
                        "font_size": 1,  # this is in *pixels* in most Omni UI contexts
                        "color": ui.color("#f5f5f5"),
                    },
                )


# NameTag Manager


class NameTagManager:
    def __init__(
        self,
        widget_cls=NameTagWidget,
        width: float = 0.15,
        height: float = 0.15,
        resolution_scale: int = 20,
        unit_to_pixel_scale: int = 40,
        z_offset: float = 0.25,
        rotation_deg_xyz: Gf.Vec3d = Gf.Vec3d(90, 0, 0),
    ):
        self._widget = None
        self._z_offset = float(z_offset)
        self._last_name: Optional[str] = None

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

        # Start at origin
        space_stack = [
            SpatialSource.new_translation_source(Gf.Vec3d(0.0, 0.0, 0.0)),
            SpatialSource.new_rotation_source(
                Gf.Vec3d(
                    math.radians(rotation_deg_xyz[0]),
                    math.radians(rotation_deg_xyz[1]),
                    math.radians(rotation_deg_xyz[2]),
                )
            ),
        ]
        self._ui_container = UiContainer(
            self._widget_component, space_stack=space_stack
        )
        self.hide()

    def show(self):
        self._widget_component.visible = True

    def hide(self):
        self._widget_component.visible = False

    def update(self, guide: "BaseGuide", highlighter: StepHighlighter):
        idx = highlighter.step_index
        if idx < 0 or idx >= len(getattr(guide, "SEQUENCE", [])):
            self.hide()
            self._last_name = None
            return

        name = guide.SEQUENCE[idx]
        if not name:
            self.hide()
            self._last_name = None
            return

        live = guide.get_live_part_pose(name)
        if not live:
            self.hide()
            return

        pos, _quat = live
        pos_above = pos + Gf.Vec3d(0.0, 0.0, self._z_offset)

        # Only update text when it changes
        if name != self._last_name:
            if self._widget and hasattr(self._widget, "set_text"):
                self._widget.set_text(name)
            self._last_name = name

        # Update position every frame
        self._ui_container.manipulator.translation = isaac_world_to_xr_ui(pos_above)
        self.show()

    def destroy(self):
        try:
            self.hide()
        except Exception:
            pass
        self._widget = None
        try:
            if hasattr(self._ui_container, "destroy"):
                self._ui_container.destroy()
        except Exception:
            pass
        self._ui_container = None
        self._widget_component = None


# Base guide


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
        total = len(self.SEQUENCE) or 1
        return [f"Step {i+1}/{total}: {name}" for i, name in enumerate(self.SEQUENCE)]

    def get_live_part_pose(self, name: str):
        p = getattr(self, "_paths", {}).get(name)
        if not p:
            return None
        return physx_get_pose(p)

    def create_highlighter(self, stage: Usd.Stage) -> VisualSequenceHighlighter:
        MaterialRegistry.ensure_all(stage)  # make sure materials exist
        hl = VisualSequenceHighlighter(
            stage, self.SEQUENCE, MaterialRegistry.visual_path
        )
        hl.set_sequence(self.SEQUENCE, start_index=0)
        return hl

    def create_physics_binder(self) -> PhysicsSequenceBinder:
        return PhysicsSequenceBinder(self.SEQUENCE, MaterialRegistry.physics_path)

    def maybe_auto_advance(self, highlighter: StepHighlighter):
        idx = highlighter.step_index
        checks: Sequence[Callable[[], bool]] | None = getattr(self, "_checks", None)
        if not checks:
            return

        if idx >= len(checks):
            return

        if checks[idx]():
            highlighter.advance()

    def any_part_fallen_below_table(
        self,
        part_names: Sequence[str],
        z_margin: float = 0.2,
    ) -> bool:
        table_pos = getattr(self, "_static_table_pos", None)
        if table_pos is None:
            return False

        table_z = float(table_pos[2])

        for name in part_names:
            live = self.get_live_part_pose(name)
            if not live:
                continue
            pos, _ = live
            if pos[2] < abs(table_z - z_margin):
                return True

        return False

    def update_ghost_visibility_for_step(self, step_index: int) -> None:
        if self._stage is None:
            return

        names_to_show: set[str] = set()

        # Show current step ghost
        if 0 <= step_index < len(self.SEQUENCE):
            current_name = self.SEQUENCE[step_index]
            if current_name:
                names_to_show.add(current_name)

        else:
            # Show ghosts for unmet constraints
            global_ok = True
            if hasattr(self, "is_final_assembly_valid"):
                try:
                    global_ok = bool(self.is_final_assembly_valid())
                except Exception:
                    global_ok = True

            if not global_ok and hasattr(self, "final_unmet_constraints"):
                try:
                    issues = list(self.final_unmet_constraints())
                except Exception:
                    issues = []

                # Show only the first failing part's ghost
                if issues:
                    names_to_show.add(issues[0][0])
                    # names_to_show.update([name for name, _ in issues])

        for name, ghost_path in self._ghost_paths_by_name.items():
            prim = self._stage.GetPrimAtPath(ghost_path)
            if not prim or not prim.IsValid():
                continue
            img = UsdGeom.Imageable(prim)
            if name in names_to_show:
                img.MakeVisible()
            else:
                img.MakeInvisible()

    def update_previews_for_step(self, highlighter: StepHighlighter) -> None:
        if not getattr(self, "enable_ghosts", True):
            return
        self.update_ghost_visibility_for_step(highlighter.step_index)

    def is_final_assembly_valid(self) -> bool:
        return True

    def final_unmet_constraints(self) -> list[tuple[str, str]]:
        return []
