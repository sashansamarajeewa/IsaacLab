from .base import BaseGuide, MaterialHighlighter

class DrawerGuide(BaseGuide):
    SEQUENCE = ["DrawerBox", "DrawerBottom", "DrawerTop"]

    def step_label(self, highlighter: MaterialHighlighter) -> str:
        idx = highlighter.step_index
        total = highlighter.total_steps or 1
        name = highlighter.current_name or "Done"
        if idx == 0:
            return f"Step 1/{total}: Pick up {name}"
        elif idx == 1:
            return f"Step 2/{total}: Move {name} into DrawerBox"
        elif idx == 2:
            return f"Step 3/{total}: Insert {name} to complete assembly"
        else:
            return "Assembly complete!"
