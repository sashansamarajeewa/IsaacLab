# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Record a single furniture-assembly demonstration using your custom teleop UI
(guide, highlighting, HUD) while saving trajectories via Isaac Lab RecorderManager.

One run = one participant + one task = one successful demo exported to HDF5.
Also logs completion time as HDF5 attribute and optionally into a CSV.
"""

import argparse
import contextlib
import os
import time
from collections.abc import Callable

from isaaclab.app import AppLauncher

# -------------------------- CLI --------------------------
parser = argparse.ArgumentParser(description="Record assembly demos with custom teleop UI")
parser.add_argument("--task", type=str, required=True, help="Name of the task")
parser.add_argument("--participant_id", type=str, required=True, help="Participant identifier, e.g. P01")
parser.add_argument("--out_dir", type=str, default="./datasets", help="Directory to save datasets")
parser.add_argument("--dataset_file", type=str, default=None, help="Optional explicit dataset file path")
# parser.add_argument("--time_log_csv", type=str, default=None, help="Optional CSV path to append completion times")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--device", type=str, default="cpu")

parser.add_argument(
    "--teleop_device",
    type=str,
    default="dualhandtracking_abs",
    help="teleop device key (e.g. dualhandtracking_abs, handtracking, keyboard, spacemouse)",
)
parser.add_argument("--num_success_steps", type=int, default=10, help="Consecutive success steps to finalize demo")
parser.add_argument("--step_hz", type=int, default=30, help="Rate limit for non-XR teleop (Hz)")

parser.add_argument("--guide", type=str, default=None)

parser.add_argument("--disable_highlight", action="store_true")
parser.add_argument("--disable_instructions", action="store_true")
parser.add_argument("--disable_ghosts", action="store_true")

parser.add_argument("--enable_pinocchio", action="store_true", default=False)

# AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

# Enable XR if handtracking
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

# -------------------------- rest imports (after app launch) --------------------------
import gymnasium as gym
import torch
import numpy as np
import omni.log
import omni.usd

from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

from guides import base, loader


class RateLimiter:
    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def _ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _default_dataset_path() -> str:
    task_short = args_cli.task.split(":")[-1].replace("/", "_")
    fname = f"{args_cli.participant_id}_{task_short}.hdf5"
    return os.path.join(args_cli.out_dir, fname)


# def _append_time_csv(csv_path: str, participant_id: str, task: str, seconds: float, success: bool):
#     if not csv_path:
#         return
#     header = "timestamp_iso,participant_id,task,success,completion_time_sec\n"
#     row = f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{participant_id},{task},{int(success)},{seconds:.6f}\n"
#     write_header = not os.path.exists(csv_path)
#     with open(csv_path, "a", encoding="utf-8") as f:
#         if write_header:
#             f.write(header)
#         f.write(row)


def annotate_hdf5(dataset_path: str, completion_time_sec: float, participant_id: str, task: str):
    """Add time + metadata as HDF5 attributes without changing dataset layout."""
    try:
        import h5py
    except Exception:
        omni.log.warn("h5py not available; skipping HDF5 attribute annotation.")
        return

    if not os.path.exists(dataset_path):
        omni.log.warn(f"Dataset file not found for annotation: {dataset_path}")
        return

    with h5py.File(dataset_path, "a") as f:
        if "data" in f:
            data_grp = f["data"]
            # pick last demo
            demo_keys = list(data_grp.keys())
            if not demo_keys:
                return
            def _demo_index(k: str) -> int:
                try:
                    return int(k.split("_")[-1])
                except Exception:
                    return -1
            demo_key = sorted(demo_keys, key=_demo_index)[-1]
            demo_grp = data_grp[demo_key]
            demo_grp.attrs["completion_time_sec"] = float(completion_time_sec)
            demo_grp.attrs["participant_id"] = str(participant_id)
            demo_grp.attrs["task"] = str(task)
        else:
            # set file-level attrs
            f.attrs["completion_time_sec"] = float(completion_time_sec)
            f.attrs["participant_id"] = str(participant_id)
            f.attrs["task"] = str(task)


def main():
    # Output paths
    _ensure_dir(args_cli.out_dir)
    dataset_path = args_cli.dataset_file or _default_dataset_path()
    dataset_dir = os.path.dirname(dataset_path)
    dataset_name_wo_ext = os.path.splitext(os.path.basename(dataset_path))[0]
    _ensure_dir(dataset_dir)

    # Parse env cfg
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.terminations.time_out = None

    # Extract success term
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    # XR compatibility
    if getattr(args_cli, "xr", False):
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    # Recorder config
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = dataset_dir
    env_cfg.recorders.dataset_filename = dataset_name_wo_ext
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    # Create env
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # Guide + UI
    stage = omni.usd.get_context().get_stage()
    guide = loader.load_guide(task_name=args_cli.task, guide_name=args_cli.guide)
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
        highlighter = guide.create_highlighter(stage)
    else:
        highlighter = DummyHighlighter(total_steps=len(getattr(guide, "SEQUENCE", [])))
        base.MaterialRegistry.ensure_all(stage)

    phys_binder = guide.create_physics_binder()

    hud = None
    if not args_cli.disable_instructions:
        hud = base.HUDManager(base.SimpleSceneWidget)
        hud.show()
        hud.update(guide, highlighter)

    # Teleop flow flags + timing
    should_reset = False
    teleoperation_active = not getattr(args_cli, "xr", False)  # XR starts inactive
    demo_started = False
    start_time = None
    success_step_count = 0
    finished = False
    completion_time_sec = None

    def reset_trial():
        nonlocal should_reset
        should_reset = True
        print("Reset requested")

    def start_teleop():
        nonlocal teleoperation_active, demo_started, start_time
        teleoperation_active = True
        if not demo_started:
            demo_started = True
            start_time = time.time()
        print("Teleoperation activated")

    def stop_teleop():
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    callbacks: dict[str, Callable[[], None]] = {
        "R": reset_trial,
        "RESET": reset_trial,
        "START": start_teleop,
        "STOP": stop_teleop,
    }

    # Optional small control panel
    control_hud = base.ControlHUD(
        base.ControlPanelWidget,
        callbacks={"start": start_teleop, "stop": stop_teleop, "reset": reset_trial},
    )
    control_hud.show()

    # Create teleop interface
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)
        else:
            omni.log.warn(f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default.")
            teleop_interface = create_teleop_device(args_cli.teleop_device, {}, callbacks)
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
    
    # Rate limiting
    rate_limiter = None if getattr(args_cli, "xr", False) else RateLimiter(args_cli.step_hz)

    # Initial reset
    env.reset()
    guide.on_reset(env)
    highlighter.refresh_after_reset()
    phys_binder.refresh_after_reset()
    guide.update_previews_for_step(highlighter)
    if hud is not None:
        hud.update(guide, highlighter)
    teleop_interface.reset()

    print(f"Recording to: {dataset_path}")
    print("Do the task. On success, demo is exported and the app exits.")

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            action = teleop_interface.advance()

            if teleoperation_active:
                actions = action.repeat(env.num_envs, 1)
                env.step(actions)
            else:
                env.sim.render()

            # Your interface updates
            guide.maybe_auto_advance(highlighter)
            guide.update_previews_for_step(highlighter)
            if hud is not None:
                hud.update(guide, highlighter)

            # Safety reset
            if guide.any_part_fallen_below_table(getattr(guide, "MOVING_PARTS", [])):
                print("Object fell below table -> reset")
                should_reset = True

            # Success detection
            if success_term is not None:
                is_success = bool(success_term.func(env, **success_term.params)[0])
            else:
                # Consider completed when guide reaches last step
                is_success = (highlighter.step_index >= highlighter.total_steps)

            if is_success:
                success_step_count += 1
                if success_step_count >= args_cli.num_success_steps:
                    # Export demo
                    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                    env.recorder_manager.set_success_to_episodes(
                        [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes([0])

                    if start_time is not None:
                        completion_time_sec = time.time() - start_time
                    else:
                        completion_time_sec = float("nan")

                    finished = True
            else:
                success_step_count = 0

            # Reset handling
            if should_reset:
                env.sim.reset()
                env.recorder_manager.reset()
                env.reset()
                guide.on_reset(env)
                highlighter.refresh_after_reset()
                phys_binder.refresh_after_reset()
                guide.update_previews_for_step(highlighter)
                if hud is not None:
                    hud.update(guide, highlighter)
                teleop_interface.reset()

                should_reset = False
                success_step_count = 0
                demo_started = False
                start_time = None

            # Stop when finished
            if finished:
                break

            if env.sim.is_stopped():
                break

            if rate_limiter:
                rate_limiter.sleep(env)

    # Cleanup + time logging
    if completion_time_sec is not None:
        # annotate HDF5
        annotate_hdf5(dataset_path, completion_time_sec, args_cli.participant_id, args_cli.task)
        # if args_cli.time_log_csv is None:
        #     # default CSV next to datasets
        #     csv_path = os.path.join(args_cli.out_dir, "timings.csv")
        # else:
        #     csv_path = args_cli.time_log_csv
        # _append_time_csv(csv_path, args_cli.participant_id, args_cli.task, completion_time_sec, success=True)
        print(f"Completion time: {completion_time_sec:.3f} sec")
        # print(f"Timing appended to: {csv_path}")

    if hud is not None:
        hud.destroy()
    control_hud.destroy()
    env.close()
    print("Done. Closing app.")


if __name__ == "__main__":
    main()
    simulation_app.close()
