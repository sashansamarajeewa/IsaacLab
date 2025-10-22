from .base import BaseGuide, VisualSequenceHighlighter
from pxr import UsdGeom, Usd, Gf
import math

def _ang_deg(q1: Gf.Quatd, q2: Gf.Quatd):
    dq = q1 * q2.GetInverse()
    a = 2.0 * math.degrees(math.acos(max(-1.0, min(1.0, abs(dq.GetReal())))))
    return min(a, 360.0 - a)

class DrawerGuide(BaseGuide):
    SEQUENCE = ["DrawerBox", "DrawerBox", "DrawerBottom", "DrawerTop"]
    
    def __init__(self):
        super().__init__()
        self._checks = [
            self._check_pickup_box,
            self._check_braced_box,
            self._check_bottom_insert,
            self._check_top_insert,
        ]
        # Cache absolute prim paths for the CURRENT env (set on reset!)
        self._paths = {}

    def on_reset(self, env):
        """Resolve prim paths once per episode."""
        env_ns = env.scene.env_ns
        names = ["Table", "ObstacleLeft", "ObstacleFront", "DrawerBox", "DrawerBottom", "DrawerTop"]
        stage = env.scene.stage
        self._paths.clear()
        for n in names:
            # Prefer exact known paths; otherwise search **only within** this env root:
            prim = stage.GetPrimAtPath(f"{env_ns}/{n}")
            if not prim or not prim.IsValid():
                # fallback: bounded search under env root (no full-stage scan)
                for p in Usd.PrimRange(stage.GetPrimAtPath(env_ns)):
                    if p.GetName() == n:
                        prim = p
                        break
            if prim and prim.IsValid():
                self._paths[n] = str(prim.GetPath())
            else:
                self._paths[n] = None

    def step_label(self, highlighter: VisualSequenceHighlighter) -> str:
        idx, total = highlighter.step_index, (highlighter.total_steps or 1)
        return (
            f"Step 1/{total}: Pick up DrawerBox" if idx == 0 else
            f"Step 2/{total}: Brace DrawerBox against left obstacle" if idx == 1 else
            f"Step 3/{total}: Insert DrawerBottom into DrawerBox" if idx == 2 else
            f"Step 4/{total}: Insert DrawerTop into DrawerBox" if idx == 3 else
            "Assembly complete!"
        )

    def maybe_auto_advance(self, env, highlighter: VisualSequenceHighlighter):
        idx = highlighter.step_index
        if idx >= len(self._checks):
            return
        stage = env.scene.stage
        # ONE cache per frame:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        if self._checks[idx](env, stage, cache):
            highlighter.advance()

    # ---------- Helpers ----------
    def _get_tf(self, stage, cache, name):
        p = self._paths.get(name)
        if not p:
            return None
        prim = stage.GetPrimAtPath(p)
        if not prim or not prim.IsValid():
            return None
        return cache.GetLocalToWorldTransform(prim)

    def _pos(self, xf): return xf.ExtractTranslation()
    def _rot(self, xf): return xf.ExtractRotation().GetQuat()

    # ---------- Checks ----------
    def _check_pickup_box(self, env, stage, cache):
        table_xf = self._get_tf(stage, cache, "Table")
        box_xf   = self._get_tf(stage, cache, "DrawerBox")
        if not (table_xf and box_xf): return False
        return (self._pos(box_xf)[2] - self._pos(table_xf)[2]) >= 0.06

    def _check_braced_box(self, env, stage, cache):
        obs_xf = self._get_tf(stage, cache, "ObstacleLeft")
        box_xf = self._get_tf(stage, cache, "DrawerBox")
        if not (obs_xf and box_xf): return False
        d = (self._pos(obs_xf) - self._pos(box_xf)).GetLength()
        ang = _ang_deg(self._rot(obs_xf), self._rot(box_xf))
        return (d <= 0.03) and (ang <= 15.0)

    def _check_bottom_insert(self, env, stage, cache):
        box_xf = self._get_tf(stage, cache, "DrawerBox")
        bot_xf = self._get_tf(stage, cache, "DrawerBottom")
        if not (box_xf and bot_xf): return False
        d = (self._pos(box_xf) - self._pos(bot_xf)).GetLength()
        ang = _ang_deg(self._rot(box_xf), self._rot(bot_xf))
        # Add a gentle vertical bias so it’s actually “inside”, not hovering
        dz = self._pos(bot_xf)[2] - self._pos(box_xf)[2]
        return (d <= 0.02) and (ang <= 12.0) and (dz <= -0.005)

    def _check_top_insert(self, env, stage, cache):
        box_xf = self._get_tf(stage, cache, "DrawerBox")
        top_xf = self._get_tf(stage, cache, "DrawerTop")
        if not (box_xf and top_xf): return False
        d = (self._pos(box_xf) - self._pos(top_xf)).GetLength()
        ang = _ang_deg(self._rot(box_xf), self._rot(top_xf))
        dz = self._pos(top_xf)[2] - self._pos(box_xf)[2]
        return (d <= 0.02) and (ang <= 12.0) and (dz >= 0.005)