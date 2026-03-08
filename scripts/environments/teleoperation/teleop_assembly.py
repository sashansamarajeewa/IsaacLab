# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable
from pathlib import Path

from isaaclab.app import AppLauncher
from llm_step_config import build_llm_checker_for_run
from PIL import Image

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Teleoperation for Isaac Lab environments."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="dualhandtracking_abs",
    help="Device for interacting with environment. Examples: keyboard, spacemouse, gamepad, handtracking, manusvive",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--sensitivity", type=float, default=1.0, help="Sensitivity factor."
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument("--guide", type=str, default=None)
parser.add_argument(
    "--disable_highlight",
    action="store_true",
    help="Disable visual part highlighting",
)
parser.add_argument(
    "--disable_instructions",
    action="store_true",
    help="Disable the instruction widget HUD",
)
parser.add_argument(
    "--disable_ghosts",
    action="store_true",
    help="Disable ghost preview objects for target poses",
)
parser.add_argument(
    "--disable_nametag",
    action="store_true",
    help="Disable the floating name tag for the current step target",
)
parser.add_argument(
    "--capture_targets",
    action="store_true",
    help="Capture and save target PNGs for each step",
)
parser.add_argument(
    "--llm_checker",
    action="store_true",
    help="Capture and save target PNGs for each step",
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

from isaaclab.devices.teleop_device_factory import create_teleop_device

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.assembly  # noqa: F401

import omni.ui as ui
import omni.ui.scene as sc
import omni.usd
from guides import base, loader


def main() -> None:
    """
    Run keyboard teleoperation with Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    env_cfg.env_name = args_cli.task
    # modify configuration
    env_cfg.terminations.time_out = None

    if args_cli.xr:
        # If cameras are not enabled and XR is enabled, remove camera configs
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs_safe(env_cfg)
            if hasattr(env_cfg.observations, "policy") and hasattr(
                env_cfg.observations.policy, "head_camera"
            ):
                env_cfg.observations.policy.head_camera = None
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
    manual_next_step_requested = False
    auto_advance_block_frames = 0
    AUTO_ADVANCE_COOLDOWN_FRAMES = 10  # ~10 frames; tweak (5–20)

    # USD stage + highlight material
    stage = omni.usd.get_context().get_stage()
    # Guide + Highlighter
    guide = loader.load_guide(task_name=args_cli.task, guide_name=args_cli.guide)

    # Configure guide options from CLI
    guide.enable_ghosts = not args_cli.disable_ghosts

    class DummyHighlighter:
        def __init__(self, total_steps: int):
            self._idx = 0
            self._total = total_steps

        def advance(self):
            if self._idx < self._total:
                self._idx += 1

        def refresh_after_reset(self):
            self._idx = 0

        @property
        def step_index(self):
            return self._idx

        @property
        def total_steps(self):
            return self._total

    if not args_cli.disable_highlight:
        # normal visual highlighter
        highlighter = guide.create_highlighter(stage)
    else:
        # no visual material binding
        highlighter = DummyHighlighter(total_steps=len(getattr(guide, "SEQUENCE", [])))

        base.MaterialRegistry.ensure_all(stage)

    total_real = len(getattr(guide, "SEQUENCE", []))
    targets_root = str(Path(__file__).resolve().parent / "targets")
    guide_folder = args_cli.guide or "default"
    capture_base_dir = Path(targets_root) / args_cli.task / guide_folder
    llm_checker = None
    if args_cli.llm_checker:
        llm_checker = build_llm_checker_for_run(
            task_name=args_cli.task,
            guide_name=args_cli.guide,
            num_steps=total_real,
            targets_root=targets_root,
        )

    # Physics binder unaffected by highlight flag
    phys_binder = guide.create_physics_binder()

    # HUD
    hud = None
    if not args_cli.disable_instructions:
        hud = base.HUDManager(base.SimpleSceneWidget)
        hud.update(guide, highlighter)

    name_tag = None
    if not args_cli.disable_nametag:
        name_tag = base.NameTagManager()

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
        
    def manual_next_step() -> None:
        nonlocal manual_next_step_requested
        manual_next_step_requested = True
        print("Manual NEXT requested from XR UI")

    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
        "NEXT": manual_next_step,
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
        if (
            hasattr(env_cfg, "teleop_devices")
            and args_cli.teleop_device in env_cfg.teleop_devices.devices
        ):
            teleop_interface = create_teleop_device(
                args_cli.teleop_device,
                env_cfg.teleop_devices.devices,
                teleoperation_callbacks,
            )
        else:
            omni.log.warn(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            # Create fallback teleop device
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, {}, teleoperation_callbacks
            )

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
    guide.on_reset(env)
    highlighter.refresh_after_reset()
    phys_binder.refresh_after_reset()
    guide.update_previews_for_step(highlighter)
    last_step_idx = None
    last_final_sig = None
    need_hud_update = False
    step_idx = highlighter.step_index
    total_real = len(getattr(guide, "SEQUENCE", []))

    if last_step_idx is None or step_idx != last_step_idx:
        need_hud_update = True
    else:
        # If we're in verification, final messages can change even if step idx doesn't
        if step_idx >= total_real and hasattr(guide, "final_unmet_constraints"):
            try:
                issues = list(guide.final_unmet_constraints())
            except Exception:
                issues = []
            sig = tuple(msg for _, msg in issues[:2])
            if sig != last_final_sig:
                last_final_sig = sig
                need_hud_update = True

    if hud is not None and need_hud_update:
        hud.update(guide, highlighter)
        last_step_idx = step_idx
        teleop_interface.reset()

    print("Teleoperation started. Press 'R' to reset the environment.")

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

                manual_advanced = False

                if manual_next_step_requested:
                    manual_next_step_requested = False
                    manual_advanced = True

                    idx = highlighter.step_index  # the step we are force-completing

                    if 0 <= idx < total_real:
                        # 1) Snap the CURRENT step's parts (your SNAP_PLAN decides what that means)
                        snap_ok = guide.snap_step_to_target(env, idx)
                        print(f"[NEXT_STEP] idx={idx} snap_ok={snap_ok}")

                        # 2) Run the step check once to trigger any side-effects
                        checks = getattr(guide, "_checks", None)
                        step_ok = False
                        if isinstance(checks, (list, tuple)) and 0 <= idx < len(checks):
                            try:
                                step_ok = bool(checks[idx]())
                            except Exception as e:
                                omni.log.warn(f"manual check failed idx={idx}: {e}")

                        # 3) If the step is now complete, run the hook (important for Desk rotation)
                        if step_ok and hasattr(guide, "on_step_completed"):
                            try:
                                guide.on_step_completed(env, idx)
                            except Exception as e:
                                omni.log.warn(f"on_step_completed failed idx={idx}: {e}")

                    # 4) Advance exactly one step
                    highlighter.advance()

                    auto_advance_block_frames = AUTO_ADVANCE_COOLDOWN_FRAMES
                    if llm_checker is not None:
                        llm_checker.reset_for_new_step()

                # Auto-advance only when not blocked and not manual on this frame
                if not manual_advanced:
                    if auto_advance_block_frames > 0:
                        auto_advance_block_frames -= 1
                    else:
                        guide.maybe_auto_advance(highlighter, env)

                if args_cli.capture_targets and args_cli.enable_cameras:
                    idx = highlighter.step_index
                    if 0 < idx <= total_real:
                        step_key = str(idx)
                        out_path = capture_base_dir / f"step_{step_key}.png"

                        # Only save once per step (don’t overwrite unless you want to)
                        if not out_path.exists():
                            cam = env.scene["head_camera"]
                            rgb = cam.data.output["rgb"][0].cpu().numpy()
                            save_rgb_png(rgb, out_path)
                
                # LLM logic
                if (not args_cli.capture_targets) and args_cli.enable_cameras and llm_checker is not None:
                    idx = highlighter.step_index
                    if 0 <= idx <= total_real:
                        cam = env.scene["head_camera"]
                        rgb = (
                            cam.data.output["rgb"][0].cpu().numpy()
                        )  # HWC uint8 (adjust if needed)

                        step_key = str(idx + 1)
                        step_text = guide.get_all_instructions()[idx]

                        if llm_checker.update(
                            step_key=step_key,
                            step_text=step_text,
                            current_rgb_uint8_hwc=rgb,
                        ):
                            highlighter.advance()
                            llm_checker.reset_for_new_step()

                guide.update_previews_for_step(highlighter)
                if hud is not None:
                    hud.update(guide, highlighter)
                if name_tag is not None:
                    name_tag.update(guide, highlighter)

                if guide.any_part_fallen_below_table(
                    getattr(guide, "MOVING_PARTS", [])
                ):
                    print("An object has fallen below table...resetting")
                    should_reset_recording_instance = True

                if should_reset_recording_instance:
                    env.reset()
                    should_reset_recording_instance = False
                    guide.on_reset(env)
                    highlighter.refresh_after_reset()
                    phys_binder.refresh_after_reset()
                    guide.update_previews_for_step(highlighter)
                    if hud is not None:
                        hud.update(guide, highlighter)
                    print("Environment reset complete")
        except Exception as e:
            omni.log.error(f"Error during simulation step: {e}")
            break

    # close the simulator
    # if hud is not None:
    #     hud.destroy()
    # if name_tag is not None:
    #     name_tag.destroy()
    env.close()
    print("Environment closed")


from typing import Any


def remove_camera_configs_safe(env_cfg: Any) -> Any:
    import logging

    logger = logging.getLogger(__name__)

    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import CameraCfg

    # Collect all cameras defined on the scene
    camera_names: list[str] = []
    for attr_name in dir(env_cfg.scene):
        try:
            attr = getattr(env_cfg.scene, attr_name)
        except Exception:
            continue
        if isinstance(attr, CameraCfg):
            camera_names.append(attr_name)

    if not camera_names:
        return env_cfg

    # Remove camera configs
    for cam_name in camera_names:
        if hasattr(env_cfg.scene, cam_name):
            delattr(env_cfg.scene, cam_name)
            logger.info(f"Removed camera config: {cam_name}")

    # Remove any ObsTerms that reference removed cameras
    if hasattr(env_cfg.observations, "policy"):
        policy = env_cfg.observations.policy

        obs_to_delete: list[str] = []
        for obs_name in dir(policy):
            try:
                obsterm = getattr(policy, obs_name)
            except Exception:
                continue

            params = getattr(obsterm, "params", None)
            if not params:
                continue

            for v in params.values():
                if isinstance(v, SceneEntityCfg) and v.name in camera_names:
                    obs_to_delete.append(obs_name)
                    break

        for obs_name in sorted(set(obs_to_delete)):
            if hasattr(policy, obs_name):
                delattr(policy, obs_name)
                logger.info(f"Removed camera observation term: {obs_name}")

    return env_cfg

def save_rgb_png(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    img.save(str(path))
    print(f"[capture_targets] Saved: {path}")

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
