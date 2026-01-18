# llm_step_config.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import json
import numpy as np
from PIL import Image

from llm_step_checker import LLMStepChecker  # the class we discussed earlier


def build_llm_checker_for_run(
    *,
    task_name: str,
    guide_name: Optional[str],
    num_steps: int,
    targets_root: str = "targets",
) -> Optional[LLMStepChecker]:
    """
    Auto-discovers:
      - targets/<task_name>/<guide_name>/config.json
      - targets/<task_name>/<guide_name>/step_<i>.png

    Only steps with existing target PNGs are loaded.
    Steps can be disabled via config.json steps[<i>].enabled=false.
    """
    guide_folder = guide_name or "default"
    base_dir = Path(targets_root) / task_name / guide_folder
    if not base_dir.exists():
        return None

    cfg: Dict[str, Any] = {}
    cfg_path = base_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Prompts + step enable flags
    base_prompt = str(cfg.get("base_prompt", "")).strip()

    steps_cfg: Dict[str, Any] = (
        cfg.get("steps", {}) if isinstance(cfg.get("steps", {}), dict) else {}
    )
    step_prompts: Dict[str, str] = {}
    enabled_steps: Dict[str, bool] = {}

    for k, v in steps_cfg.items():
        step_key = str(k)
        if isinstance(v, dict):
            if "prompt" in v and isinstance(v["prompt"], str):
                step_prompts[step_key] = v["prompt"]
            if "enabled" in v:
                enabled_steps[step_key] = bool(v["enabled"])

    llm = LLMStepChecker(
        model=str(cfg.get("model", "gpt-5.2")),
        period_s=float(cfg.get("period_s", 1.0)),
        consecutive_required=int(cfg.get("consecutive_required", 2)),
        max_image_side=int(cfg.get("max_image_side", 512)),
        base_prompt=base_prompt,
        step_prompts=step_prompts,
        enabled_steps=enabled_steps,
    )

    found_any = False
    for i in range(1, num_steps + 1):
        step_key = str(i)

        # If explicitly disabled, skip loading target
        if step_key in enabled_steps and not enabled_steps[step_key]:
            continue

        png = base_dir / f"step_{i}.png"
        if not png.exists():
            continue

        llm.set_target_png_path(step_key, str(png))
        found_any = True

    return llm if found_any else None
