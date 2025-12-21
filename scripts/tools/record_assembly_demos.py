# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Record furniture-assembly demonstrations using your custom teleop UI
(guide, highlighting, HUD) while saving trajectories via Isaac Lab RecorderManager.

Default: one run = one participant + one task = 1 successful demo exported to HDF5.
Supports multiple demos per run via --num_demos.
Stores completion time per demo as HDF5 attributes under /data/demo_<i>.
"""

import argparse
import contextlib
import os
import time
from collections.abc import Callable

from isaaclab.app import AppLauncher

# -------------------------- CLI --------------------------
parser = argparse.ArgumentParser(
    description="Record assembly demos with custom teleop UI"
)
parser.add_argument("--task", type=str, required=True, help="Name of the task")
parser.add_argument(
    "--participant_id", type=str, required=True, help="Participant identifier, e.g. P01"
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="/workspace/isaaclab/datasets",
    help="Directory to save datasets",
)
parser.add_argument(
    "--dataset_file", type=str, default=None, help="Optional explicit dataset file path"
)

parser.add_argument("--num_envs", type=int, default=1)

parser.add_argument(
    "--teleop_device",
    type=str,
    default="dualhandtracking_abs",
    help="teleop device key (e.g. dualhandtracking_abs, handtracking, keyboard, spacemouse)",
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Consecutive success steps to finalize demo",
)
parser.add_argument(
    "--step_hz", type=int, default=30, help="Rate limit for non-XR teleop (Hz)"
)

parser.add_argument(
    "--num_demos", type=int, default=1, help="Number of successful demos to record"
)

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
    import pinocchio

# Enable XR if handtracking
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

# -------------------------- imports after app launch --------------------------
import gymnasium as gym
import torch
import omni.log
import omni.usd

from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

from scripts.environments.teleoperation.guides import base, loader


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


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def default_dataset_path() -> str:
    task_short = args_cli.task.split(":")[-1].replace("/", "_")
    fname = f"{args_cli.participant_id}_{task_short}.hdf5"
    return os.path.join(args_cli.out_dir, fname)


def annotate_hdf5_demo(
    dataset_path: str,
    demo_index: int,
    completion_time_sec: float,
    participant_id: str,
    task: str,
):

    try:
        import h5py
    except Exception:
        omni.log.warn("h5py not available; skipping HDF5 attribute annotation.")
        return

    if not os.path.exists(dataset_path):
        omni.log.warn(f"Dataset file not found for annotation: {dataset_path}")
        return

    with h5py.File(dataset_path, "a") as f:
        if "data" not in f:
            omni.log.warn(
                "No '/data' group found in dataset. Writing file-level attrs instead."
            )
            f.attrs["completion_time_sec"] = float(completion_time_sec)
            f.attrs["participant_id"] = str(participant_id)
            f.attrs["task"] = str(task)
            f.attrs["demo_index"] = int(demo_index)
            return

        data_grp = f["data"]
        demo_key = f"demo_{int(demo_index)}"
        if demo_key not in data_grp:
            omni.log.warn(
                f"Expected demo group '{demo_key}' not found under '/data'. Available: {list(data_grp.keys())}"
            )
            # fallback to writing file-level attrs
            f.attrs["completion_time_sec"] = float(completion_time_sec)
            f.attrs["participant_id"] = str(participant_id)
            f.attrs["task"] = str(task)
            f.attrs["demo_index"] = int(demo_index)
            return

        demo_grp = data_grp[demo_key]
        demo_grp.attrs["completion_time_sec"] = float(completion_time_sec)
        demo_grp.attrs["participant_id"] = str(participant_id)
        demo_grp.attrs["task"] = str(task)


def reset_all(env, guide, highlighter, phys_binder, hud, teleop_interface):
    """Reset sim/env/recorder?UI/guide state for the next demo."""
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


def main():
    # Output paths
    ensure_dir(args_cli.out_dir)
    dataset_path = args_cli.dataset_file or default_dataset_path()
    dataset_dir = os.path.dirname(dataset_path)
    dataset_name_wo_ext = os.path.splitext(os.path.basename(dataset_path))[0]
    ensure_dir(dataset_dir)

    # Parse env cfg
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    env_cfg.env_name = args_cli.task
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    # Extract success term
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    if args_cli.xr:
        # If cameras are not enabled and XR is enabled, remove camera configs
        if not args_cli.enable_cameras:
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

    # Guide and UI
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

    # Teleop flow flags and timing
    should_reset = False
    teleoperation_active = not getattr(args_cli, "xr", False)  # XR starts inactive
    demo_started = False
    start_time = None
    success_step_count = 0

    demos_recorded = 0
    finished = False

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

    # Create teleop interface
    try:
        if (
            hasattr(env_cfg, "teleop_devices")
            and args_cli.teleop_device in env_cfg.teleop_devices.devices
        ):
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks
            )
        else:
            omni.log.warn(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, {}, callbacks
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

    # Rate limiting
    rate_limiter = (
        None if getattr(args_cli, "xr", False) else RateLimiter(args_cli.step_hz)
    )

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
    if args_cli.num_demos == 0:
        print(
            "Do the task. On each success, a demo is exported. (num_demos=0 => infinite)"
        )
    else:
        print(f"Do the task. Need {args_cli.num_demos} successful demo(s).")

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
                print("Object fell below table. Reset")
                should_reset = True

            # Success detection
            step_complete = highlighter.step_index >= highlighter.total_steps
            global_ok = (
                guide.is_final_assembly_valid()
                if hasattr(guide, "is_final_assembly_valid")
                else step_complete
            )
            is_success = step_complete and global_ok

            # If success stable long enough, export one demo
            if is_success:
                success_step_count += 1
                if success_step_count >= args_cli.num_success_steps:
                    # Determine the demo index that will be written
                    prev_count = env.recorder_manager.exported_successful_episode_count

                    # Export demo
                    env.recorder_manager.record_pre_reset(
                        [0], force_export_or_skip=False
                    )
                    env.recorder_manager.set_success_to_episodes(
                        [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes([0])

                    # Compute time per demo
                    if start_time is not None:
                        completion_time_sec = time.time() - start_time
                    else:
                        completion_time_sec = float("nan")

                    # Annotate the demo group exported
                    annotate_hdf5_demo(
                        dataset_path=dataset_path,
                        demo_index=prev_count,  # newly exported demo index
                        completion_time_sec=completion_time_sec,
                        participant_id=args_cli.participant_id,
                        task=args_cli.task,
                    )

                    demos_recorded += 1
                    print(
                        f"Demo {demos_recorded} exported. Time: {completion_time_sec:.3f} sec"
                    )

                    # Check stop condition
                    if args_cli.num_demos > 0 and demos_recorded >= args_cli.num_demos:
                        finished = True
                    else:
                        # Prepare next demo
                        reset_all(
                            env, guide, highlighter, phys_binder, hud, teleop_interface
                        )

                        # Reset per demo
                        success_step_count = 0
                        demo_started = False
                        start_time = None

                        if getattr(args_cli, "xr", False):
                            teleoperation_active = False

            else:
                success_step_count = 0

            # Manual reset handling
            if should_reset:
                reset_all(env, guide, highlighter, phys_binder, hud, teleop_interface)

                should_reset = False
                success_step_count = 0
                demo_started = False
                start_time = None
                if getattr(args_cli, "xr", False):
                    teleoperation_active = False

            # Stop when finished
            if finished:
                break

            if env.sim.is_stopped():
                break

            if rate_limiter:
                rate_limiter.sleep(env)

    # Cleanup
    if hud is not None:
        hud.destroy()
    env.close()
    print("Done. Closing app.")


if __name__ == "__main__":
    main()
    simulation_app.close()
