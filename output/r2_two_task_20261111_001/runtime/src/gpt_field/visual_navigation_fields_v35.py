"""Keep anatomical and wrist-bearing objectives separate from scene geometry.

Both fields learn only from acknowledged observations. Prefer a currently
qualified anatomical objective; otherwise a wrist-bearing objective may propose
noncontact navigation in its observed subspace. Neither estimates target depth,
changes a physical budget, or authorizes a command.
"""
from __future__ import annotations

import copy

from .visual_objective_v32 import VisualGoalFieldV32
from .visual_wrist_goal_v35 import WristBearingGoalFieldV35


class VisualNavigationFieldsV35(VisualGoalFieldV32):
    def __init__(self):
        self.wrist_navigation = WristBearingGoalFieldV35()
        super().__init__()

    def configure(self, state8, monitor, prediction, *, joint_position=None):
        super().configure(state8, monitor, prediction, joint_position=joint_position)
        self.wrist_navigation.configure(state8, monitor, prediction,
                                        joint_position=joint_position)
        return self.summary()

    def observe(self, before8, action7, after8, *, monitor, prediction,
                joint_position=None, acknowledged=True):
        super().observe(before8, action7, after8, monitor=monitor,
                        prediction=prediction, joint_position=joint_position,
                        acknowledged=acknowledged)
        self.wrist_navigation.observe(before8, action7, after8, monitor=monitor,
            prediction=prediction, joint_position=joint_position,
            acknowledged=acknowledged)
        return self.summary()

    def propose(self, state8, **kwargs):
        anatomy = super().propose(state8, **kwargs)
        wrist = self.wrist_navigation.propose(state8, **kwargs)
        source = 'anatomical_relative_pixels' if anatomy.get('accepted') else 'wrist_bearing'
        selected = anatomy if anatomy.get('accepted') else wrist
        result = copy.deepcopy(selected)
        result['objective_source'] = source
        result['available_objectives'] = {
            'anatomical_relative_pixels': {'accepted': anatomy.get('accepted'),
                                         'reason': anatomy.get('reason')},
            'wrist_bearing': {'accepted': wrist.get('accepted'),
                             'reason': wrist.get('reason')},
        }
        for row in result.get('candidates', []):
            row['objective_source'] = source
            row['contact_alignment_evidence'] = False
            row['target_depth_estimated'] = False
        return result

    def summary(self):
        result = super().summary()
        result['navigation_objectives_version'] = 'separate-anatomy-and-wrist-bearing-v35'
        result['wrist_navigation'] = self.wrist_navigation.summary()
        result['wrist_is_anatomical_or_contact_evidence'] = False
        return result
