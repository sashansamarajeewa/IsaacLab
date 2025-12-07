from typing import Optional
from .base import BaseGuide
from .drawer import DrawerGuide
from .lamp import LampGuide
from .wedge import WedgeGuide
from .hexagon import HexagonGuide
from .motor import MotorGuide

# Map task names and --guide flag to guide classes
_TASK_MAP = {
    "Isaac-Assembly-Drawer-GR1T2-Abs-v0": DrawerGuide,
    "Isaac-Assembly-Lamp-GR1T2-Abs-v0": LampGuide,
    "Isaac-Assembly-Wedge-GR1T2-Abs-v0": WedgeGuide,
    "Isaac-Assembly-Hexagon-GR1T2-Abs-v0": HexagonGuide,
    "Isaac-Assembly-Motor-GR1T2-Abs-v0": MotorGuide,
}

_GUIDE_MAP = {
    "drawer": DrawerGuide,
    "lamp": LampGuide,
    "wedge": WedgeGuide,
    "hexagon": HexagonGuide,
    "motor": MotorGuide,
}

def load_guide(task_name: Optional[str] = None, guide_name: Optional[str] = None) -> BaseGuide:
    if guide_name:
        cls = _GUIDE_MAP.get(guide_name.lower())
        if cls:
            return cls()
    if task_name and task_name in _TASK_MAP:
        return _TASK_MAP[task_name]()
    # Fallback default
    return DrawerGuide()
