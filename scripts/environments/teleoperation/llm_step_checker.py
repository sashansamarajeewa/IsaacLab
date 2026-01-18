# llm_step_checker.py
from __future__ import annotations

import base64
import io
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Dict

import numpy as np
from pydantic import BaseModel
from openai import OpenAI
from PIL import Image


class StepDecision(BaseModel):
    step_complete: bool
    confidence: float
    failure_mode: str
    reason: str


def _rgb_to_data_url(rgb_uint8_hwc: np.ndarray, max_side: int = 512) -> str:
    img = Image.fromarray(rgb_uint8_hwc, mode="RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class LLMStepChecker:
    """Rate-limited, async LLM judge that compares CURRENT vs TARGET for a given step index."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.2",
        period_s: float = 1.0,
        consecutive_required: int = 2,
        max_image_side: int = 512,
        base_prompt: str = "",
        step_prompts: Optional[Dict[str, str]] = None,
        enabled_steps: Optional[Dict[str, bool]] = None,
    ) -> None:
        self.model = model
        self.period_s = float(period_s)
        self.k = int(consecutive_required)
        self.max_image_side = int(max_image_side)

        self.base_prompt = base_prompt.strip()
        self.step_prompts = step_prompts or {}
        self.enabled_steps = enabled_steps or {}

        self._t_last = 0.0
        self._streak = 0
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._inflight: Optional[Future[StepDecision]] = None

        # step_key -> data URL
        self._target_data_url: Dict[str, str] = {}

    def set_target_image(self, step_key: str, target_rgb_uint8_hwc: np.ndarray) -> None:
        self._target_data_url[step_key] = _rgb_to_data_url(
            target_rgb_uint8_hwc, max_side=self.max_image_side
        )

    def set_target_png_path(self, step_key: str, png_path: str) -> None:
        img = Image.open(png_path).convert("RGB")
        rgb = np.array(img, dtype=np.uint8)
        self.set_target_image(step_key, rgb)

    def reset_for_new_step(self) -> None:
        self._streak = 0
        self._inflight = None
        self._t_last = 0.0

    def _is_step_enabled(self, step_key: str) -> bool:
        if step_key in self.enabled_steps:
            return bool(self.enabled_steps[step_key])
        return True

    def _submit_request(
        self,
        step_key: str,
        step_text: str,
        current_url: str,
        target_url: str,
    ) -> Future[StepDecision]:
        system = (
            "You are a strict visual inspector for a robot assembly task. "
            "Compare CURRENT vs TARGET for the given step and decide whether the step is complete. "
            "If you cannot judge due to occlusion/ambiguity, mark it as not complete."
        )

        step_hint = self.step_prompts.get(step_key, "").strip()
        user_text = "\n".join(
            s
            for s in [
                f"Step label: {step_text}",
                (f"Additional step criteria: {step_hint}" if step_hint else ""),
                (f"Global criteria: {self.base_prompt}" if self.base_prompt else ""),
                "Output JSON only with keys: step_complete (bool), confidence (0..1), failure_mode, reason.",
                "Allowed failure_mode: none, position, orientation, occluded, unknown.",
            ]
            if s
        )

        def _call() -> StepDecision:
            try:
                # Create client inside thread (safer)
                client = OpenAI()

                resp = client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": user_text},
                                {"type": "input_text", "text": "TARGET reference:"},
                                {"type": "input_image", "image_url": target_url},
                                {"type": "input_text", "text": "CURRENT frame:"},
                                {"type": "input_image", "image_url": current_url},
                            ],
                        },
                    ],
                    text_format=StepDecision,
                )

                dec = resp.output_parsed  # type: ignore
                if dec is None:
                    return StepDecision(
                        step_complete=False,
                        confidence=0.0,
                        failure_mode="unknown",
                        reason="output_parsed_none",
                    )
                return dec

            except Exception as e:
                print("\n[LLMStepChecker] Exception inside _call():", repr(e))
                print(traceback.format_exc())

                # Debug fallback: show raw output_text
                try:
                    client = OpenAI()
                    raw = client.responses.create(
                        model=self.model,
                        input=[
                            {"role": "system", "content": system},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": user_text},
                                    {"type": "input_text", "text": "TARGET reference:"},
                                    {"type": "input_image", "image_url": target_url},
                                    {"type": "input_text", "text": "CURRENT frame:"},
                                    {"type": "input_image", "image_url": current_url},
                                ],
                            },
                        ],
                    )
                    print("[LLMStepChecker] Raw output_text:\n", raw.output_text)
                except Exception as e2:
                    print("[LLMStepChecker] Raw fallback failed:", repr(e2))

                return StepDecision(
                    step_complete=False,
                    confidence=0.0,
                    failure_mode="unknown",
                    reason=f"exception:{type(e).__name__}",
                )

        return self._executor.submit(_call)

    def update(
        self,
        *,
        step_key: str,  # "1", "2", ...
        step_text: str,
        current_rgb_uint8_hwc: np.ndarray,
        min_confidence: float = 0.75,
    ) -> bool:
        """Non-blocking. Returns True when you should advance."""
        if step_key not in self._target_data_url:
            return False
        if not self._is_step_enabled(step_key):
            return False

        now = time.monotonic()

        # Consume finished request
        if self._inflight is not None and self._inflight.done():
            try:
                dec = self._inflight.result()
            except Exception:
                dec = StepDecision(
                    step_complete=False,
                    confidence=0.0,
                    failure_mode="unknown",
                    reason="request_failed",
                )
            self._inflight = None

            conf = float(getattr(dec, "confidence", 0.0))
            if conf > 1.0:  # normalize if it came back as 0..100
                conf = conf / 100.0

            fm = str(getattr(dec, "failure_mode", "unknown")).strip().lower()
            if fm in ("ok", "success", "complete", "completed"):
                fm = "none"

            ok = bool(getattr(dec, "step_complete", False)) and conf >= float(min_confidence) and fm == "none"
            self._streak = (self._streak + 1) if ok else 0
            return self._streak >= self.k

        # Rate limit and only one inflight
        if self._inflight is None and (now - self._t_last) >= self.period_s:
            self._t_last = now
            cur_url = _rgb_to_data_url(current_rgb_uint8_hwc, max_side=self.max_image_side)
            tgt_url = self._target_data_url[step_key]
            self._inflight = self._submit_request(step_key, step_text, cur_url, tgt_url)

        return False
