"""Explicit candidate role choice; paid evidence is not anatomical identity."""
from __future__ import annotations

import base64
import io
from PIL import Image, ImageDraw

from .visual_reference_candidates_v30 import CausalReferenceCandidatesV30


class CausalReferenceCandidatesV32(CausalReferenceCandidatesV30):
    version = 'role-explicit-paid-reference-review-v32'

    def summary(self):
        result = super().summary()
        result['selection_requires'] = (
            'Explicitly identify either actual tool anatomy or a hand/wrist feature rigidly '
            'attached to the EEF as a transit-only motion proxy. A proximal moving link, '
            'shadow or background is not identified by motion. Proxy use requires a separate '
            'actual-hand point/ROI and unchanged whole-hand clearance envelope. The proxy '
            'point is not a pinch point, contact target or calibrated hand projection.')
        result['candidate_role_options'] = ['tool_anatomy', 'rigid_transit_proxy']
        result['role_selection_is_model_claim'] = True
        result['proxy_defines_anatomical_goal_or_clearance'] = False
        return result

    def image(self, observation=None):
        value = super().image(observation)
        if value is None:
            return None
        board = Image.open(io.BytesIO(base64.b64decode(value.split(',', 1)[1]))).convert('RGB')
        draw = ImageDraw.Draw(board)
        w, h = self.current.rgb.shape[1], self.current.rgb.shape[0]
        draw.rectangle((w*2, 0, w*4, 31), fill=(24, 24, 28))
        draw.text((w*2+8, 8), 'CURRENT RGB: explicitly identify feature ROLE', fill='white')
        draw.rectangle((0, h*2+50, w*4, h*2+75), fill=(24, 24, 28))
        draw.text((8, h*2+56), 'Rigid EEF proxy is transit-only; independently track actual hand. No motion credit/identity proof.', fill='white')
        stream = io.BytesIO(); board.save(stream, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()
