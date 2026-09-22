"""Bind selective policies and independent test reports to exact model evidence."""
import copy

from .adapt_calibration import metric_report, probability_records, validate_calibration
from .adapt_data import fingerprint, question_fingerprint
from .adapt_eval import verify_prediction_dataset
from .risk import DEFAULT_THRESHOLDS, evaluate_risk_policy, select_risk_policy


def validate_bound_policy(policy, calibration, questions, *, require_passed=False):
    """Validate links; artifact authenticity still requires a trusted outer hash."""
    if (not isinstance(policy, dict) or type(policy.get("schema_version")) is not int
            or policy["schema_version"] != 1 or policy.get("stage") != "bound_policy"):
        raise ValueError("expected a versioned bound policy")
    if not isinstance(policy.get("binding"), dict):
        raise ValueError("policy is missing its model/data/schema binding")
    validate_calibration(calibration, binding=policy["binding"])
    if (policy.get("calibration_sha256") != fingerprint(calibration)
            or policy.get("binding", {}).get("questions_sha256") != question_fingerprint(questions)):
        raise ValueError("policy calibration or ordered question schema differs")
    qid, selection = policy.get("question_id"), policy.get("selection")
    if not isinstance(qid, str) or qid not in questions or not isinstance(selection, dict):
        raise ValueError("policy needs a known question and selection report")
    question = questions[qid]
    count = 2 if question["type"] == "noul" else len(question["criteria"])
    if (type(selection.get("schema_version")) is not int or selection["schema_version"] != 1
            or selection.get("stage") != "selection" or selection.get("n_options") != count
            or type(selection.get("passed")) is not bool):
        raise ValueError("policy selection does not match the question's option schema")
    if require_passed and selection["passed"] is not True:
        raise ValueError("policy did not qualify; do not open the final test for this candidate")


def select_bound_policy(predictions, calibration, dataset_dir, *, question_id, required_languages,
                        thresholds=DEFAULT_THRESHOLDS, max_error=.05, min_coverage=.5, alpha=.05,
                        declaration=None):
    """Select only on the policy partition and freeze model/calibration/schema links.

    Alpha must already account for any model-candidate comparisons in the caller's
    predeclared experiment. The inner selector covers thresholds, languages and
    error/coverage bounds. Declaration stores that protocol's trusted identifier.
    """
    questions = verify_prediction_dataset(predictions, dataset_dir)
    if predictions["split"] != "policy":
        raise ValueError("select thresholds only on the policy partition")
    samples = probability_records(predictions, questions, calibration, question_id=question_id)
    if not samples:
        raise ValueError("policy partition has no evidence for this question")
    selection = select_risk_policy(samples, required_languages=required_languages, thresholds=thresholds,
                                   max_error=max_error, min_coverage=min_coverage, alpha=alpha)
    policy = {"schema_version": 1, "stage": "bound_policy", "question_id": question_id,
              "binding": copy.deepcopy(predictions["binding"]), "calibration_sha256": fingerprint(calibration),
              "policy_predictions_sha256": fingerprint(predictions), "selection": selection,
              "declaration": copy.deepcopy(declaration)}
    validate_bound_policy(policy, calibration, questions)
    return policy


def evaluate_bound_policy(predictions, policy, calibration, dataset_dir):
    """Test a locked, qualifying policy on fresh final-test representatives once."""
    questions = verify_prediction_dataset(predictions, dataset_dir)
    if predictions["split"] != "test":
        raise ValueError("final policy evaluation requires the test partition")
    validate_bound_policy(policy, calibration, questions, require_passed=True)
    if fingerprint(predictions["binding"]) != fingerprint(policy["binding"]):
        raise ValueError("final model, data or rendering differs from the selected policy")
    samples = probability_records(predictions, questions, calibration, question_id=policy["question_id"])
    report = evaluate_risk_policy(samples, policy["selection"])
    return {"schema_version": 1, "stage": "bound_final_test", "policy_sha256": fingerprint(policy),
            "calibration_sha256": fingerprint(calibration), "binding": copy.deepcopy(predictions["binding"]),
            "test_predictions_sha256": fingerprint(predictions), "evaluation": report,
            "metrics": {"raw": metric_report(predictions, questions),
                        "calibrated": metric_report(predictions, questions, calibration)}}
