from __future__ import annotations

import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Dict

import numpy as np
from pydantic import BaseModel, Field
from openai import OpenAI
from PIL import Image


class StepDecision(BaseModel):
    step_complete: bool
    confidence: float = Field(ge=0.0, le=1.0)
    failure_mode: str  # "none" | "position" | "orientation" | "occluded" | "unknown"
    reason: str


def _rgb_to_data_url(rgb_uint8_hwc: np.ndarray, max_side: int = 512) -> str:
    """Encode RGB uint8 HWC to PNG data URL; downscale to reduce cost/latency."""
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
    """
    Rate-limited, async LLM judge:
      - compares current frame to a target/reference frame
      - returns 'advance' only after K consecutive positives
    """

    def __init__(
        self,
        *,
        model: str = "gpt-5.2",
        period_s: float = 0.8,
        consecutive_required: int = 2,
        max_image_side: int = 512,
    ) -> None:
        self.client = OpenAI()
        self.model = model
        self.period_s = float(period_s)
        self.k = int(consecutive_required)
        self.max_image_side = int(max_image_side)

        self._t_last = 0.0
        self._streak = 0
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._inflight: Optional[Future[StepDecision]] = None

        # Cache target images as data URLs to avoid repeated encoding
        self._target_data_url: Dict[str, str] = {}

    def set_target_image(self, step_key: str, target_rgb_uint8_hwc: np.ndarray) -> None:
        self._target_data_url[step_key] = _rgb_to_data_url(
            target_rgb_uint8_hwc, max_side=self.max_image_side
        )

    def reset_for_new_step(self) -> None:
        self._streak = 0
        self._inflight = None
        self._t_last = 0.0

    def _submit_request(self, step_text: str, current_url: str, target_url: str) -> Future[StepDecision]:
        system = (
            "You are a strict visual inspector for a robot assembly task. "
            "Decide if the CURRENT frame matches the TARGET reference for the described step. "
            "Answer ONLY using the provided JSON schema."
        )

        user_text = (
            f"Step: {step_text}\n"
            "Compare CURRENT vs TARGET. If you cannot judge due to occlusion or ambiguity, "
            'set failure_mode="occluded" or "unknown" and step_complete=false.'
        )

        def _call() -> StepDecision:
            # Structured Outputs via SDK parse (Pydantic)
            resp = self.client.responses.parse(
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
            return resp.output_parsed  # type: ignore

        return self._executor.submit(_call)

    def update(
        self,
        *,
        step_key: str,
        step_text: str,
        current_rgb_uint8_hwc: np.ndarray,
    ) -> bool:
        """
        Returns True when you should advance.
        Non-blocking: may return False while request is inflight.
        """
        if step_key not in self._target_data_url:
            return False

        now = time.monotonic()

        # If a request finished, consume it
        if self._inflight is not None and self._inflight.done():
            try:
                dec = self._inflight.result()
            except Exception:
                dec = StepDecision(
                    step_complete=False, confidence=0.0, failure_mode="unknown", reason="request_failed"
                )

            self._inflight = None

            ok = bool(dec.step_complete) and float(dec.confidence) >= 0.75 and dec.failure_mode == "none"
            self._streak = (self._streak + 1) if ok else 0

            return self._streak >= self.k

        # Rate limit + only one inflight request
        if self._inflight is None and (now - self._t_last) >= self.period_s:
            self._t_last = now
            cur_url = _rgb_to_data_url(current_rgb_uint8_hwc, max_side=self.max_image_side)
            tgt_url = self._target_data_url[step_key]
            self._inflight = self._submit_request(step_text, cur_url, tgt_url)

        return False
