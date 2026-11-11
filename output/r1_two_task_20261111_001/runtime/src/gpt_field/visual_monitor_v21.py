"""Existing causal response/attachment contracts with bounded RGB maintenance."""
from .visual_monitor_v12 import RoleAwareVisualMonitorV12
from .visual_reference_track_v21 import PartFeatureTrackV21


class RoleAwareVisualMonitorV21(RoleAwareVisualMonitorV12):
    feature_bundle_class = PartFeatureTrackV21

    def summary(self):
        result = super().summary()
        result["version"] = "causal-response-bounded-appearance-v21"
        return result
