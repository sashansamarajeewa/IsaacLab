import functools
import omni.ui as ui
import omni.ui.scene as sc
from omni.kit.xr.scene_view.utils.ui_container import UiContainer
from omni.kit.xr.scene_view.utils.manipulator_components.widget_component import WidgetComponent
from omni.kit.xr.scene_view.utils.spatial_source import SpatialSource
from pxr import Gf

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


def on_widget_constructed(widget_instance, text):
    print("Widget is ready!")
    widget_instance.label.text = text
    widget_instance.invalidate()

# Show instruction function (self-less functional style)
def show_instruction(text):
    print("show_instruction")
    widget_component = WidgetComponent(
        SimpleSceneWidget,
        width=6,
        height=1,
        resolution_scale=10,
        unit_to_pixel_scale=20,
        update_policy=sc.Widget.UpdatePolicy.ON_DEMAND,
        construct_callback=functools.partial(on_widget_constructed, text=text)
    )

    # define spatial sources
    space_stack = [
        SpatialSource.new_translation_source(Gf.Vec3d(0, 1.5, 2)),
        SpatialSource.new_look_at_camera_source()
    ]

    # create UiContainer
    UiContainer(widget_component, space_stack=space_stack)
