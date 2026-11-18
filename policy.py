"""Set the course quality gates before running a new evaluation window."""

MIN_ROC_AUC = 0.51
MAX_ROC_AUC_DROP = 0.03
MIN_ROC_AUC_IMPROVEMENT = 0.02
DRIFT_THRESHOLD = 0.20


def needs_retraining(champion_auc, baseline_auc):
    """Retrain on low current quality or degradation from the approval window."""
    return champion_auc < MIN_ROC_AUC or baseline_auc - champion_auc >= MAX_ROC_AUC_DROP


def passes_promotion(candidate_auc, champion_auc=None):
    """Require an absolute floor and, after bootstrap, an improvement on the same data."""
    return candidate_auc >= MIN_ROC_AUC and (
        champion_auc is None or candidate_auc - champion_auc >= MIN_ROC_AUC_IMPROVEMENT
    )
