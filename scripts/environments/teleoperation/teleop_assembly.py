import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="dualhandtracking_abs", help="Device for interacting with environment")
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

app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


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
from pxr import Gf

def pre_process_actions(
    teleop_data: tuple[np.ndarray, bool] | list[tuple[np.ndarray, np.ndarray, np.ndarray]], num_envs: int, device: str
) -> torch.Tensor:
    """Convert teleop data to the format expected by the environment action space.

    Args:
        teleop_data: Data from the teleoperation device.
        num_envs: Number of environments.
        device: Device to create tensors on.

    Returns:
        Processed actions as a tensor.
    """
    # compute actions based on environment
    (left_wrist_pose, right_wrist_pose, hand_joints) = teleop_data[0]
    # Reconstruct actions_arms tensor with converted positions and rotations
    actions = torch.tensor(
        np.concatenate([
            left_wrist_pose,  # left ee pose
            right_wrist_pose,  # right ee pose
            hand_joints,  # hand joint angles
        ]),
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    # Concatenate arm poses and hand joint angles
    return actions

def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    # modify configuration
    env_cfg.terminations.time_out = None

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance():
        """Reset the environment to its initial state.

        This callback is triggered when the user presses the reset key (typically 'R').
        It's useful when:
        - The robot gets into an undesirable configuration
        - The user wants to start over with the task
        - Objects in the scene need to be reset to their initial positions

        The environment will be reset on the next simulation step.
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    def start_teleoperation():
        """Activate teleoperation control of the robot.

        This callback enables active control of the robot through the input device.
        It's typically triggered by a specific gesture or button press and is used when:
        - Beginning a new teleoperation session
        - Resuming control after temporarily pausing
        - Switching from observation mode to control mode

        While active, all commands from the device will be applied to the robot.
        """
        nonlocal teleoperation_active
        teleoperation_active = True

    def stop_teleoperation():
        """Deactivate teleoperation control of the robot.

        This callback temporarily suspends control of the robot through the input device.
        It's typically triggered by a specific gesture or button press and is used when:
        - Taking a break from controlling the robot
        - Repositioning the input device without moving the robot
        - Pausing to observe the scene without interference

        While inactive, the simulation continues to render but device commands are ignored.
        """
        nonlocal teleoperation_active
        teleoperation_active = False

    # create controller
    # Create GR1T2 retargeter with desired configuration
    gr1t2_retargeter = GR1T2Retargeter(
        enable_visualization=True,
        num_open_xr_hand_joints=2 * (int(OpenXRSpec.HandJointEXT.XR_HAND_JOINT_LITTLE_TIP_EXT) + 1),
        device=env.unwrapped.device,
        hand_joint_names=env.scene["robot"].data.joint_names[-22:],
    )

    # Create hand tracking device with retargeter
    teleop_interface = OpenXRDevice(
        env_cfg.xr,
        retargeters=[gr1t2_retargeter],
    )
    teleop_interface.add_callback("RESET", reset_recording_instance)
    teleop_interface.add_callback("START", start_teleoperation)
    teleop_interface.add_callback("STOP", stop_teleoperation)

    # Hand tracking needs explicit start gesture to activate
    teleoperation_active = False

    # add teleoperation key for env reset (for all devices)
    teleop_interface.add_callback("R", reset_recording_instance)
    print(teleop_interface)

    # reset environment
    env.reset()
    teleop_interface.reset()
    
    # initialize InstructionDisplayManager


    # define Scene Widget
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

    # callback to capture widget instance when it's constructed
    widget_ref = {}

    def on_widget_constructed(widget_instance):
        print("Widget is ready!")
        widget_ref["widget"] = widget_instance

    # create WidgetComponent
    widget_component = WidgetComponent(
        SimpleSceneWidget,
        width=5,
        height=1.5,
        resolution_scale=10,
        unit_to_pixel_scale=20,
        update_policy=sc.Widget.UpdatePolicy.ON_DEMAND,
        construct_callback=on_widget_constructed
    )

    # define spatial sources
    space_stack = [
        SpatialSource.new_translation_source(Gf.Vec3d(0, 1.5, 2)),
        SpatialSource.new_look_at_camera_source()
    ]

    # create UiContainer
    ui_container = UiContainer(widget_component, space_stack=space_stack)

    # display initial instruction after widget is ready
    if "widget" in widget_ref:
         widget_ref["widget"].label.text = "Step 1: Pick up Drawer Base Item"


    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get device command
            teleop_data = teleop_interface.advance()

            # Only apply teleop commands when active
            if teleoperation_active:
                # compute actions based on environment
                actions = pre_process_actions(teleop_data, env.num_envs, env.device)
                # apply actions
                env.step(actions)
            else:
                env.sim.render()

            if should_reset_recording_instance:
                env.reset()
                should_reset_recording_instance = False

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

