import argparse
import math

from isaaclab.app import AppLauncher

# -------- CLI --------
parser = argparse.ArgumentParser(description="Teleoperation with step-by-step highlighting.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="dualhandtracking_abs", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument("--enable_pinocchio", action="store_true", default=False, help="Enable Pinocchio.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401
app_launcher_args["xr"] = True

# -------- Launch app --------
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


# -------- Std deps --------
import gymnasium as gym
import numpy as np
import torch
import omni.log

from isaacsim.xr.openxr import OpenXRSpec
from isaaclab.devices import OpenXRDevice
if args_cli.enable_pinocchio:
    from isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1t2_retargeter import GR1T2Retargeter
    import isaaclab_tasks.manager_based.manipulation.assembly  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import omni.ui as ui
import omni.ui.scene as sc
from omni.kit.xr.scene_view.utils.ui_container import UiContainer
from omni.kit.xr.scene_view.utils.manipulator_components.widget_component import WidgetComponent
from omni.kit.xr.scene_view.utils.spatial_source import SpatialSource

from pxr import Gf, Usd, UsdShade, Sdf
import omni.usd


# -----------------------------------------------------------------------------
# Teleop action pre-processing
# -----------------------------------------------------------------------------
def pre_process_actions(
    teleop_data: tuple[np.ndarray, bool] | list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_envs: int,
    device: str,
) -> torch.Tensor:
    (left_wrist_pose, right_wrist_pose, hand_joints) = teleop_data[0]
    actions = torch.tensor(
        np.concatenate([left_wrist_pose, right_wrist_pose, hand_joints]),
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    return actions


# -----------------------------------------------------------------------------
# Generic prim discovery by name
# -----------------------------------------------------------------------------
def find_prim_paths_by_name(stage: Usd.Stage, name_token: str) -> list[str]:
    """Return prim paths for all prims whose leaf name == name_token (handles env_0, env_1, ...)."""
    return [p.GetPath().pathString for p in stage.Traverse() if p.GetName() == name_token]


# -----------------------------------------------------------------------------
# USD Material create/bind
# -----------------------------------------------------------------------------
def create_preview_surface_material(
    stage: Usd.Stage,
    prim_path: str,
    diffuse=(1.0, 0.9, 0.2),
    emissive=(8.0, 8.0, 1.0),
    roughness=0.6,
    metallic=0.0,
) -> UsdShade.Material:
    """Create or get a USD PreviewSurface material at prim_path and return the Material object."""
    mat_prim = stage.DefinePrim(prim_path, "Material")
    material = UsdShade.Material(mat_prim)

    shader_path = f"{prim_path}/Shader"
    shader_prim = stage.DefinePrim(shader_path, "Shader")
    shader = UsdShade.Shader(shader_prim)
    shader.CreateIdAttr("UsdPreviewSurface")

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)

    surface_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface_output)
    return material


class MaterialSwap:
    """Save/restore original material bindings and bind a provided USD material on demand."""
    def __init__(self, stage: Usd.Stage):
        self._stage = stage
        self._saved: dict[str, UsdShade.Material | None] = {}

    def _get_bound_material(self, prim: Usd.Prim):
        db = UsdShade.MaterialBindingAPI(prim).GetDirectBinding()
        return db.GetMaterial()

    def apply(self, prim_path: str, material: UsdShade.Material, stronger_than_descendants: bool = True):
        """Bind material to prim; by default as stronger-than-descendants."""
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim not found: {prim_path}")

        if prim_path not in self._saved:
            self._saved[prim_path] = self._get_bound_material(prim)

        strength = (
            UsdShade.Tokens.strongerThanDescendants
            if stronger_than_descendants
            else UsdShade.Tokens.weakerThanDescendants
        )
        UsdShade.MaterialBindingAPI(prim).Bind(
            material,
            bindingStrength=strength,
            materialPurpose=UsdShade.Tokens.allPurpose,
        )

    def restore(self, prim_path: str):
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return
        prev = self._saved.pop(prim_path, None)
        api = UsdShade.MaterialBindingAPI(prim)
        if prev:
            api.Bind(prev, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        else:
            api.UnbindDirectBinding()


class MaterialHighlighter:
    """Generic, name-based highlighter that works across multiple env instances."""
    def __init__(self, stage: Usd.Stage, material: UsdShade.Material):
        self.stage = stage
        self.material = material
        self._swap = MaterialSwap(stage)
        self._sequence: list[str] = []
        self._step: int = 0
        self._current_name: str | None = None
        self._current_paths: list[str] = []

    def set_sequence(self, names: list[str], start_index: int = 0):
        self._sequence = list(names)
        self._step = max(0, min(start_index, len(self._sequence) - 1))
        self.highlight_name(self._sequence[self._step] if self._sequence else None)

    def highlight_name(self, name: str | None):
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
        """Re-apply highlight after a reset (paths may change)."""
        name = self._current_name
        if not name:
            return
        self.highlight_name(name)

    @property
    def current_name(self) -> str | None:
        return self._current_name

    @property
    def step_index(self) -> int:
        return self._step

    @property
    def total_steps(self) -> int:
        return len(self._sequence)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Parse config and make env
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.terminations.time_out = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    should_reset_recording_instance = False
    teleoperation_active = True

    # USD stage + highlight material
    stage = omni.usd.get_context().get_stage()
    highlight_mat_path = "/World/Looks/Highlight_Gold"
    highlight_material = create_preview_surface_material(
        stage,
        prim_path=highlight_mat_path,
        diffuse=(1.0, 0.9, 0.0),
        emissive=(10.0, 8.0, 0.0),
    )

    # Assembly order (editable)
    TARGET_SEQUENCE = ["DrawerBox", "DrawerBottom", "DrawerTop"]

    # Highlighter
    highlighter = MaterialHighlighter(stage, highlight_material)
    highlighter.set_sequence(TARGET_SEQUENCE, start_index=0)  # starts at DrawerBox

    # Callbacks
    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    def start_teleoperation():
        nonlocal teleoperation_active
        teleoperation_active = True

    def stop_teleoperation():
        nonlocal teleoperation_active
        teleoperation_active = False

    def clear_highlight():
        highlighter.clear()

    def next_target():
        highlighter.advance()
        if widget_ref.get("widget"):
            idx = highlighter.step_index
            total = highlighter.total_steps
            name = highlighter.current_name or "Done"
            widget_ref["widget"].label.text = f"Step {min(idx+1, total)}/{total}: Pick up {name}"

    # Retargeter + device
    gr1t2_retargeter = GR1T2Retargeter(
        enable_visualization=True,
        num_open_xr_hand_joints=2 * (int(OpenXRSpec.HandJointEXT.XR_HAND_JOINT_LITTLE_TIP_EXT) + 1),
        device=env.unwrapped.device,
        hand_joint_names=env.scene["robot"].data.joint_names[-22:],
    )
    teleop_interface = OpenXRDevice(env_cfg.xr, retargeters=[gr1t2_retargeter])
    teleop_interface.add_callback("RESET", reset_recording_instance)
    teleop_interface.add_callback("START", start_teleoperation)
    teleop_interface.add_callback("STOP", stop_teleoperation)
    teleop_interface.add_callback("R", reset_recording_instance)
    teleop_interface.add_callback("C", clear_highlight)
    teleop_interface.add_callback("N", next_target)  # Manual step advance for now
    teleoperation_active = False
    print(teleop_interface)

    # First reset and initial highlight
    env.reset()
    teleop_interface.reset()
    highlighter.refresh_current()

    # Simple HUD
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

    widget_ref = {}
    def on_widget_constructed(widget_instance):
        widget_ref["widget"] = widget_instance
        idx = highlighter.step_index
        total = highlighter.total_steps
        name = highlighter.current_name or "Done"
        widget_instance.label.text = f"Step {min(idx+1, total)}/{total}: Pick up {name}"

    widget_component = WidgetComponent(
        SimpleSceneWidget,
        width=4,
        height=1.3,
        resolution_scale=10,
        unit_to_pixel_scale=20,
        update_policy=sc.Widget.UpdatePolicy.ON_DEMAND,
        construct_callback=on_widget_constructed,
    )
    space_stack = [
        SpatialSource.new_translation_source(Gf.Vec3d(0, 1.5, 2)),
        SpatialSource.new_rotation_source(Gf.Vec3d(math.radians(90), math.radians(0), math.radians(0))),
    ]
    ui_container = UiContainer(widget_component, space_stack=space_stack)

    # Main loop
    while simulation_app.is_running():
        with torch.inference_mode():
            teleop_data = teleop_interface.advance()

            if teleoperation_active:
                actions = pre_process_actions(teleop_data, env.num_envs, env.device)
                env.step(actions)
            else:
                env.sim.render()

            if should_reset_recording_instance:
                env.reset()
                teleop_interface.reset()
                should_reset_recording_instance = False

                # Re-apply current highlight (env prim paths may change)
                highlighter.refresh_current()
                if widget_ref.get("widget"):
                    idx = highlighter.step_index
                    total = highlighter.total_steps
                    name = highlighter.current_name or "Done"
                    widget_ref["widget"].label.text = f"Step {min(idx+1, total)}/{total}: Pick up {name}  (C=clear, N=next)"

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
