"""V5 bounded persistence, retaining V4's hard geometric and measured checks."""
from .visual_field_v4 import MetricFieldExecutorV4, validate_metric_intent


class MetricFieldExecutorV5(MetricFieldExecutorV4):
    """An intent may last up to 120 actions, never beyond its spatial bounds.

    Arrival, obstacles, unknown regions, joint limits, motion disagreement and
    the measured travel budget still stop it early. This is not an episode cap.
    """
    @staticmethod
    def _validate_intent(intent):
        return validate_metric_intent(intent, action_limit=120)

    def summary(self):
        result = super().summary()
        if self.intent is not None:
            result["version"] = "metric-hypothesis-distance-executor-v5"
        return result
