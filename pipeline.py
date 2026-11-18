"""Evaluate a dataset release or roll back the approved production model."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import mlflow
from mlflow.exceptions import MlflowException

from data import DataError, VERSIONS, fetch_and_track, validate_splits
from model import compare_distributions, evaluate, reference_profile, train_model
import policy
from registry import (
    APPROVAL_TAG, APPROVED, DEFAULT_EXPERIMENT, configure_tracking, get_champion, load_version,
    move_champion, record_candidate, register_approved,
)

PROFILE_ARTIFACT = "training-profile.json"


def _log_scores(prefix, scores):
    mlflow.log_metrics({f"{prefix}_{name}": value for name, value in scores.items()})


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_release(client, name, release, workspace, result):
    paths, fingerprints = fetch_and_track(release, Path(".exercise-data") / release,
                                          workspace / "manifests")
    result["dvc"] = fingerprints
    mlflow.log_params({f"dvc_{split}_md5": md5 for split, md5 in fingerprints.items()})
    train, evaluation = validate_splits(paths)
    result["validation"] = "passed"
    mlflow.set_tag("validation", "passed")
    mlflow.log_metrics({"train_rows": len(train), "evaluation_rows": len(evaluation)})
    mlflow.log_params({
        "min_roc_auc": policy.MIN_ROC_AUC,
        "max_roc_auc_drop": policy.MAX_ROC_AUC_DROP,
        "min_roc_auc_improvement": policy.MIN_ROC_AUC_IMPROVEMENT,
        "drift_threshold": policy.DRIFT_THRESHOLD,
    })

    champion = get_champion(client, name)
    champion_version = str(champion.version) if champion else None
    result["champion_before"] = result["champion_after"] = champion_version
    if champion is None and release != "v1":
        raise ValueError("Run v1 to approve the initial champion before evaluating a later window")

    champion_auc = None
    if champion:
        if int(champion.tags["training_last_source_row"]) >= evaluation["source_row"].min():
            raise DataError("This evaluation window is older than the champion's training cutoff")
        scores = evaluate(load_version(champion), evaluation)
        _log_scores("champion", scores)
        result["champion_metrics"] = scores
        champion_auc = scores["roc_auc"]
        baseline = float(champion.tags["approval_roc_auc"])
        result["champion_baseline_roc_auc"] = baseline
        mlflow.log_metrics({"champion_baseline_roc_auc": baseline, "roc_auc_drop": baseline - champion_auc})
        profile_path = client.download_artifacts(champion.tags["approval_run_id"], PROFILE_ARTIFACT)
        profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        drift = compare_distributions(profile, evaluation, policy.DRIFT_THRESHOLD)
        _write_json(workspace / "drift.json", drift)
        result["drift_flags"] = [feature for feature, stats in drift.items() if stats["flagged"]]
        mlflow.log_metrics({"drift_flag_count": len(result["drift_flags"]),
                            "max_feature_drift": max(stats["total_variation"] for stats in drift.values())})
        if not policy.needs_retraining(champion_auc, baseline):
            result["decision"] = "retain_no_retraining"
            return

    result["retrained"] = True
    estimator = train_model(train)
    scores = evaluate(estimator, evaluation)
    _log_scores("candidate", scores)
    result["candidate_metrics"] = scores
    run_id = result["run_id"]
    profile_path = workspace / PROFILE_ARTIFACT
    _write_json(profile_path, reference_profile(train))
    mlflow.log_artifact(str(profile_path))
    model_uri = record_candidate(estimator, train, release, scores, run_id)
    result["candidate_model_uri"] = model_uri
    if not policy.passes_promotion(scores["roc_auc"], champion_auc):
        result["decision"] = "retain_rejected" if champion else "bootstrap_rejected"
        return

    target = register_approved(
        client, name, model_uri, run_id, release, int(train["source_row"].max()), scores["roc_auc"],
    )
    result["candidate_version"] = target
    move_champion(client, name, champion_version, target)
    result["champion_after"] = target
    result["decision"] = "promoted" if champion else "bootstrapped"


def roll_back(client, name, requested_version, result):
    champion = get_champion(client, name)
    if champion is None:
        raise ValueError("Approve an initial champion before requesting a rollback")
    prior = str(champion.version)
    result["champion_before"] = result["champion_after"] = prior
    result["rollback_version"] = requested_version
    version = client.get_model_version(name, requested_version)
    if version.tags.get(APPROVAL_TAG) != APPROVED:
        raise ValueError("Rollback requires a previously approved model version")
    if prior != requested_version:
        # Verify the stored artifact can still be loaded before exposing it to inference.
        load_version(version)
        move_champion(client, name, prior, requested_version)
    result["champion_after"] = requested_version
    result["decision"] = "rolled_back" if prior != requested_version else "rollback_unchanged"


def execute(args):
    client, name = configure_tracking()
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT))
    release = args.dataset_version
    operation = "pipeline" if release else "rollback"
    output = Path(os.getenv("RUN_OUTPUT_DIR", ".exercise-results"))
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="bank-results-") as tmp, mlflow.start_run(run_name=release or operation) as run:
        workspace = Path(tmp)
        result = {
            "operation": operation, "dataset_version": release, "run_id": run.info.run_id,
            "model_name": name, "validation": "pending" if release else "not_applicable",
            "retrained": False, "champion_before": None, "champion_after": None,
        }
        mlflow.set_tags({"operation": operation, "dataset_version": release or "not_applicable",
                         "model_name": name})
        failed = False
        try:
            if release:
                run_release(client, name, release, workspace, result)
            else:
                roll_back(client, name, args.rollback, result)
        except (ValueError, MlflowException, OSError, KeyError, RuntimeError) as exc:
            if isinstance(exc, DataError):
                result["validation"] = "failed"
            result["decision"] = "failed"
            # MLflow transport errors may contain endpoint URLs. Keep details at the source.
            if isinstance(exc, MlflowException):
                result["error"] = f"MLflow request failed ({exc.error_code})"
            else:
                result["error"] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            failed = True
        finally:
            mlflow.set_tags({
                "decision": result.get("decision", "failed"),
                "validation": result["validation"],
                "retrained": str(result["retrained"]).lower(),
                **{field: result.get(field) or "none"
                   for field in ("champion_before", "champion_after", "candidate_version")},
            })
            _write_json(workspace / "decision.json", result)
            for filename in ("drift.json", "decision.json"):
                path = workspace / filename
                if path.exists():
                    shutil.copyfile(path, output / filename)
                    mlflow.log_artifact(str(path))
                else:
                    (output / filename).unlink(missing_ok=True)
            manifests = workspace / "manifests"
            if manifests.is_dir():
                shutil.copytree(manifests, output / "manifests", dirs_exist_ok=True)
                mlflow.log_artifacts(str(manifests), artifact_path="manifests")
            if failed or result.get("decision") == "bootstrap_rejected":
                mlflow.end_run(status="FAILED")
            print(json.dumps(result, indent=2, sort_keys=True))
    return int(failed or result.get("decision") == "bootstrap_rejected")


def _version_number(value):
    if not value.isascii() or not value.isdecimal() or int(value) < 1:
        raise argparse.ArgumentTypeError("model versions are positive integers")
    return str(int(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dataset-version", choices=VERSIONS)
    action.add_argument("--rollback", type=_version_number, metavar="VERSION")
    try:
        return execute(parser.parse_args())
    except (ValueError, MlflowException, OSError) as exc:
        message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print(f"Pipeline error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
