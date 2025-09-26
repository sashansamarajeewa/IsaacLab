# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
from collections.abc import Callable

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="dualhandtracking_abs",
    help="Device for interacting with environment. Examples: keyboard, spacemouse, gamepad, handtracking, manusvive",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
    # GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import numpy as np
import torch

import omni.log

from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.assembly  # noqa: F401

import omni.ui as ui
import omni.ui.scene as sc
from omni.kit.xr.scene_view.utils.ui_container import UiContainer
from omni.kit.xr.scene_view.utils.manipulator_components.widget_component import WidgetComponent
from omni.kit.xr.scene_view.utils.spatial_source import SpatialSource

from pxr import Gf, Usd, UsdShade, Sdf
import omni.usd

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

def main() -> None:
    """
    Run keyboard teleoperation with Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    # modify configuration
    env_cfg.terminations.time_out = None

    if args_cli.xr:
        # External cameras are not supported with XR teleop
        # Check for any camera configs and disable them
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True
    
    # USD stage + highlight material
    stage = omni.usd.get_context().get_stage()
    highlight_mat_path = "/World/Looks/Highlight"
    highlight_material = create_preview_surface_material(
        stage,
        prim_path=highlight_mat_path,
        diffuse=(0.6, 0.8, 0.1),
        emissive=(0, 0, 0),
    )

    # Assembly order
    TARGET_SEQUENCE = ["DrawerBox", "DrawerBottom", "DrawerTop"]

    # Highlighter
    highlighter = MaterialHighlighter(stage, highlight_material)
    highlighter.set_sequence(TARGET_SEQUENCE, start_index=0)  # starts at DrawerBox

    # Callback handlers
    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """
        Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """
        Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")
        
    def clear_highlight():
        highlighter.clear()

    def next_target():
        highlighter.advance()
        if widget_ref.get("widget"):
            idx = highlighter.step_index
            total = highlighter.total_steps
            name = highlighter.current_name or "Done"
            widget_ref["widget"].label.text = f"Step {min(idx+1, total)}/{total}: Pick up {name}"


    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
        "C": clear_highlight,
        "N": next_target,
    }

    # For hand tracking devices, add additional callbacks
    if args_cli.xr:
        # Default to inactive for hand tracking
        teleoperation_active = False
    else:
        # Always active for other devices
        teleoperation_active = True

    # Create teleop device from config if present, otherwise create manually
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            omni.log.warn(f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default.")
            # Create fallback teleop device
            sensitivity = args_cli.sensitivity
            omni.log.error(f"Unsupported teleop device: {args_cli.teleop_device}")
            omni.log.error("Supported devices: keyboard, spacemouse, gamepad, handtracking")
            env.close()
            simulation_app.close()
            return

    except Exception as e:
        omni.log.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        omni.log.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    # reset environment
    env.reset()
    teleop_interface.reset()

    print("Teleoperation started. Press 'R' to reset the environment.")

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
    
    # simulate environment
    while simulation_app.is_running():
        try:
            # run everything in inference mode
            with torch.inference_mode():
                # get device command
                action = teleop_interface.advance()

                # Only apply teleop commands when active
                if teleoperation_active:
                    # process actions
                    actions = action.repeat(env.num_envs, 1)
                    # apply actions
                    env.step(actions)
                else:
                    env.sim.render()

                if should_reset_recording_instance:
                    env.reset()
                    should_reset_recording_instance = False
                    # Re-apply current highlight (env prim paths may change)
                    highlighter.refresh_current()
                    if widget_ref.get("widget"):
                        idx = highlighter.step_index
                        total = highlighter.total_steps
                        name = highlighter.current_name or "Done"
                        widget_ref["widget"].label.text = f"Step {min(idx+1, total)}/{total}: Pick up {name}"
                    print("Environment reset complete")
        except Exception as e:
            omni.log.error(f"Error during simulation step: {e}")
            break

    # close the simulator
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
