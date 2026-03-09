from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Dict, Any
import copy

import httpx
import numpy as np
from pydantic import BaseModel, Field
from PIL import Image


class StepDecision(BaseModel):
    step_complete: bool
    confidence: float = Field(ge=0.0, le=1.0)
    failure_mode: str  # "none" | "position" | "orientation" | "occluded" | "unknown"
    reason: str


def _rgb_to_data_url(rgb_uint8_hwc: np.ndarray, max_side: int = 512) -> str:
    """Encode uint8 RGB image to PNG data URL. (Responses API accepts base64 data URLs.)"""
    img = Image.fromarray(rgb_uint8_hwc.astype(np.uint8), mode="RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _extract_output_text(resp_json: dict) -> str:
    """
    Responses API returns output items; text is typically inside:
      output[*].content[*] where type == 'output_text'
    """
    out_chunks: list[str] = []
    for item in resp_json.get("output", []) or []:
        content = item.get("content", []) or []
        for c in content:
            if c.get("type") == "output_text":
                t = c.get("text")
                if isinstance(t, str):
                    out_chunks.append(t)
    return "\n".join(out_chunks).strip()


def _safe_json_loads(text: str) -> dict:
    """
    Try strict JSON. If model wraps JSON in text, extract first {...} block.
    (Structured Outputs should already be strict, but keep a guard.)
    """
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


# NEW: OpenAI strict json_schema requires additionalProperties=false for object schemas.
def _make_openai_strict_json_schema(schema: dict) -> dict:
    """
    Ensure additionalProperties=false on all object schemas (required by strict json_schema).
    Also ensures 'required' includes all properties keys (safe for strict mode).
    """
    schema = copy.deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            # Object schema can be expressed either by type=="object" or by having "properties"
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False

                props = node.get("properties")
                if isinstance(props, dict) and props:
                    req = set(node.get("required", []))
                    req.update(props.keys())
                    node["required"] = list(req)

            for v in node.values():
                visit(v)

        elif isinstance(node, list):
            for v in node:
                visit(v)

    visit(schema)
    return schema


class LLMStepChecker:
    """
    Rate-limited, async LLM judge that compares CURRENT vs TARGET for a given step.

    Uses Responses API image input (input_image/image_url) and Structured Outputs via text.format.
    """

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
        api_key: Optional[str] = None,
        endpoint: str = "https://api.openai.com/v1/responses",
        timeout_s: float = 30.0,
        use_json_schema: bool = True,
    ) -> None:
        self.model = model
        self.period_s = float(period_s)
        self.k = int(consecutive_required)
        self.max_image_side = int(max_image_side)
        self.base_prompt = base_prompt.strip()
        self.step_prompts = step_prompts or {}
        self.enabled_steps = enabled_steps or {}

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set in the environment (or passed to LLMStepChecker)."
            )

        self.endpoint = endpoint
        self.timeout_s = float(timeout_s)
        self.use_json_schema = bool(use_json_schema)

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
            "You are a strict visual inspector for a robot assembly task."
            "Compare CURRENT vs TARGET for the given step and decide whether the step is complete."
            "Judge object-state completion, not exact robot hand pose."
            "If you cannot judge due to occlusion or ambiguity, mark it as not complete."
        )

        step_hint = self.step_prompts.get(step_key, "").strip()

        user_text = "\n".join(
            s
            for s in [
                f"Step label: {step_text}",
                (f"Additional step criteria: {step_hint}" if step_hint else ""),
                (f"Global criteria: {self.base_prompt}" if self.base_prompt else ""),
                "Decide if CURRENT matches TARGET closely enough to count the step as complete.",
                "Return ONLY JSON that matches the requested schema.",
            ]
            if s
        )

        # Structured Outputs config (json_schema) OR fallback JSON mode
        if self.use_json_schema:
            raw_schema = StepDecision.model_json_schema()
            strict_schema = _make_openai_strict_json_schema(raw_schema)
            text_cfg = {
                "format": {
                    "type": "json_schema",
                    "name": "step_decision",
                    "strict": True,
                    "schema": strict_schema,
                }
            }
        else:
            text_cfg = {"format": {"type": "json_object"}}

        payload = {
            "model": self.model,
            "input": [
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
            "text": text_cfg,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        def _call() -> StepDecision:
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    r = client.post(self.endpoint, headers=headers, json=payload)

                if r.status_code >= 400:
                    err_text = r.text
                    print(f"[LLMStepChecker] HTTP {r.status_code}: {err_text}")

                    # Try JSON mode once if strict schema fails for any reason
                    payload["text"] = {"format": {"type": "json_object"}}
                    with httpx.Client(timeout=self.timeout_s) as client:
                        r2 = client.post(self.endpoint, headers=headers, json=payload)
                    if r2.status_code >= 400:
                        print(
                            f"[LLMStepChecker] Fallback HTTP {r2.status_code}: {r2.text}"
                        )
                        return StepDecision(
                            step_complete=False,
                            confidence=0.0,
                            failure_mode="unknown",
                            reason=f"http_error:{r2.status_code}",
                        )
                    resp_json = r2.json()
                else:
                    resp_json = r.json()

                out_text = _extract_output_text(resp_json)
                print(out_text)
                if not out_text:
                    return StepDecision(
                        step_complete=False,
                        confidence=0.0,
                        failure_mode="unknown",
                        reason="empty_output_text",
                    )

                data = _safe_json_loads(out_text)
                return StepDecision.model_validate(data)

            except Exception as e:
                print("\n[LLMStepChecker] Exception inside _call():", repr(e))
                print(traceback.format_exc())
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
        step_key: str,
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

        if self._inflight is not None and self._inflight.done():
            dec = self._inflight.result()
            self._inflight = None

            conf = float(dec.confidence)
            fm = str(dec.failure_mode).strip().lower()

            ok = (
                bool(dec.step_complete)
                and conf >= float(min_confidence)
                and fm in ("none", "")
            )
            self._streak = (self._streak + 1) if ok else 0
            print(f"streak:{self._streak}")
            return self._streak >= self.k

        if self._inflight is None and (now - self._t_last) >= self.period_s:
            self._t_last = now
            cur_url = _rgb_to_data_url(
                current_rgb_uint8_hwc, max_side=self.max_image_side
            )
            tgt_url = self._target_data_url[step_key]
            self._inflight = self._submit_request(step_key, step_text, cur_url, tgt_url)

        return False
