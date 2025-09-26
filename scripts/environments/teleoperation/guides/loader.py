from typing import Optional, Type
from .base import BaseGuide
from .drawer import DrawerGuide

# Map canonical task names (or substrings) and --guide flag to guide classes
_TASK_MAP = {
    "Isaac-Assembly-Drawer-GR1T2-Abs-v0": DrawerGuide,
}

_GUIDE_MAP = {
    "drawer": DrawerGuide,
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
