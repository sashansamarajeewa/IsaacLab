from typing import Optional
from .base import BaseGuide
from .drawer import DrawerGuide
from .lamp import LampGuide
from .blocks import BlocksGuide

# Map task names and --guide flag to guide classes
_TASK_MAP = {
    "Isaac-Assembly-Drawer-GR1T2-Abs-v0": DrawerGuide,
    "Isaac-Assembly-Lamp-GR1T2-Abs-v0": LampGuide,
    "Isaac-Assembly-Blocks-GR1T2-Abs-v0": BlocksGuide,
}

_GUIDE_MAP = {
    "drawer": DrawerGuide,
    "lamp": LampGuide,
    "blocks": BlocksGuide,
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
