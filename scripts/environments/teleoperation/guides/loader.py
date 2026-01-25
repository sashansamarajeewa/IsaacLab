from typing import Optional
from .base import BaseGuide
from .drawer import DrawerGuide
from .chair import ChairGuide
from .stool import StoolGuide
from .desk import DeskGuide

# Map task names and --guide flag to guide classes
_TASK_MAP = {
    "Isaac-Assembly-Drawer-GR1T2-Abs-v0": DrawerGuide,
    "Isaac-Assembly-Chair-GR1T2-Abs-v0": ChairGuide,
    "Isaac-Assembly-Stool-GR1T2-Abs-v0": StoolGuide,
    "Isaac-Assembly-Desk-GR1T2-Abs-v0": DeskGuide,
}

_GUIDE_MAP = {
    "drawer": DrawerGuide,
    "chair": ChairGuide,
    "stool": StoolGuide,
    "desk": DeskGuide,
}


def load_guide(
    task_name: Optional[str] = None, guide_name: Optional[str] = None
) -> BaseGuide:
    if guide_name:
        cls = _GUIDE_MAP.get(guide_name.lower())
        if cls:
            return cls()
    if task_name and task_name in _TASK_MAP:
        return _TASK_MAP[task_name]()
    # Fallback default
    return DrawerGuide()
