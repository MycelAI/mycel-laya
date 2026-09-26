"""Measure a fixed CPU deployment and compare its observed outcomes across hosts."""
import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from laya.adapt_bundle import _runtime_contract, load_bundle
from laya.adapt_data import _json_value, fingerprint
from laya.adapt_train import _atomic_save


def process_memory():
    """Return process resident-memory measurements in bytes, without new packages."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_memory = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_memory.restype = wintypes.BOOL
        if not get_memory(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"resident_bytes": counters.WorkingSetSize, "peak_resident_bytes": counters.PeakWorkingSetSize,
                "source": "GetProcessMemoryInfo; process lifetime peak working set"}
    if sys.platform.startswith("linux"):
        import resource

        fields = Path("/proc/self/statm").read_text().split()
        return {"resident_bytes": int(fields[1]) * os.sysconf("SC_PAGE_SIZE"),
                "peak_resident_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "source": "/proc/self/statm and getrusage; process lifetime peak RSS"}
    raise ValueError("memory measurement currently supports Windows and Linux")


def validate_probe(probe):
    if (not isinstance(probe, dict) or type(probe.get("schema_version")) is not int
            or probe["schema_version"] != 1 or probe.get("stage") != "deployment_probe"
            or not isinstance(probe.get("cases"), list) or not probe["cases"]):
        raise ValueError("expected a versioned nonempty deployment probe")
    for field, minimum in (("warmup", 0), ("repeats", 1)):
        if type(probe.get(field)) is not int or probe[field] < minimum:
            raise ValueError(f"probe {field} must be an integer >= {minimum}")
    seen = set()
    for case in probe["cases"]:
        if (not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"].strip()
                or case["id"] in seen or not isinstance(case.get("language"), str) or not case["language"].strip()
                or "state" not in case or case.get("expect") not in ("prediction", "review")):
            raise ValueError("probe cases need unique IDs, declared languages, states and explicit expectations")
        seen.add(case["id"])
        if case["expect"] == "review" and case.get("reason") not in (
                "unsupported_language", "empty_or_invalid_state", "unsupported_input"):
            raise ValueError("guard probes must declare a structural review reason")
        _json_value(case["state"], "state")
    if not any(case["expect"] == "prediction" for case in probe["cases"]):
        raise ValueError("include a prediction case; guard-only timing is not model serving latency")
    return probe


def _check_outcome(case, result):
    if case["expect"] == "review":
        valid = result.get("status") == "review" and result.get("value") is None and result.get("reason") == case["reason"]
    else:
        valid = ((result.get("status") == "automate" and result.get("reason") == "qualified_policy"
                  and isinstance(result.get("prediction"), dict))
                 or (result.get("status") == "review" and result.get("reason") == "below_threshold"
                     and isinstance(result.get("suggestion"), dict)))
    if not valid:
        raise ValueError(f"probe {case['id']} returned an unexpected outcome: {result.get('reason')}")


def measure(bundle_dir, expected_manifest_sha256, probe, output, *, context):
    """Time verified load and complete predict calls, separately from fast guard paths.

    Run in a fresh process. Reported peak memory is process lifetime RSS/working
    set, including imports and model loading; it is not the model's incremental
    footprint. Inputs have no gold labels and cannot establish model accuracy.
    """
    validate_probe(probe)
    output, bundle_dir = Path(output), Path(bundle_dir)
    if output.exists():
        raise FileExistsError("benchmark output already exists; preserve the earlier measurement")
    if output.resolve().is_relative_to(bundle_dir.resolve()):
        raise ValueError("write benchmark results outside the immutable bundle")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("describe host conditions and any concurrent work in context")
    implementation = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    runtime = _runtime_contract()
    before = process_memory()
    began = time.perf_counter()
    bundle = load_bundle(bundle_dir, expected_manifest_sha256=expected_manifest_sha256)
    load_ms = (time.perf_counter() - began) * 1000
    # The loader verifies this manifest against the supplied trusted checksum.
    manifest = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    if manifest["mode"] != "qualified":
        raise ValueError("review-only bundles do not measure neural serving; use a qualified bundle")
    after_load = process_memory()
    results = []
    for case in probe["cases"]:
        for _ in range(probe["warmup"]):
            _check_outcome(case, bundle.predict(case["state"], language=case["language"]))
        samples, outcomes = [], []
        for _ in range(probe["repeats"]):
            began = time.perf_counter()
            result = bundle.predict(case["state"], language=case["language"])
            samples.append((time.perf_counter() - began) * 1000)
            _check_outcome(case, result)
            outcomes.append(result)
        results.append({"id": case["id"], "case_sha256": fingerprint(case), "expect": case["expect"],
                        "samples_ms": samples, "p50_ms": float(np.percentile(samples, 50)),
                        "p95_ms": float(np.percentile(samples, 95)), "outcomes": outcomes,
                        "stable_outcomes": all(outcome == outcomes[0] for outcome in outcomes)})
    if (runtime != _runtime_contract()
            or implementation != hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()):
        raise ValueError("benchmark implementation or inference runtime changed during measurement")
    report = {"schema_version": 1, "stage": "deployment_measurement", "created_utc": datetime.now(timezone.utc).isoformat(),
              "bundle_manifest_sha256": bundle.manifest_sha256, "probe_sha256": fingerprint(probe),
              "implementation_sha256": implementation, "runtime": runtime, "context": context,
              "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
                       "processor": platform.processor(), "logical_cpus": os.cpu_count()},
              "warmup_per_case": probe["warmup"], "repeats_per_case": probe["repeats"], "verified_load_ms": load_ms,
              "memory": {"before_load": before, "after_load": after_load, "after_prediction": process_memory()},
              "cases": results,
              "scope": "CPU serial wall time: predict includes validation, rendering, tokenization, forward, calibration and policy; load includes hash verification; Python/import startup excluded from timing; OS disk cache not controlled; no quality inference from probes"}
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save(report, output)
    return report


def compare(left, right, *, probability_atol=1e-6):
    """Compare observed decisions and probabilities, not host latency equality."""
    if (isinstance(probability_atol, bool) or not isinstance(probability_atol, (int, float))
            or not math.isfinite(probability_atol) or not 0 <= probability_atol <= 1):
        raise ValueError("probability_atol must be a finite tolerance in [0, 1]")
    for report in (left, right):
        if (not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 1
                or report.get("stage") != "deployment_measurement" or not isinstance(report.get("cases"), list)
                or not report["cases"] or not isinstance(report.get("runtime"), dict)):
            raise ValueError("expected nonempty deployment measurement reports")
        for field in ("bundle_manifest_sha256", "probe_sha256", "implementation_sha256"):
            digest = report.get(field)
            if not isinstance(digest, str) or len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
                raise ValueError(f"measurement needs a valid {field}")
        repeats = report.get("repeats_per_case")
        if type(repeats) is not int or repeats < 1:
            raise ValueError("measurement repeat count must be a positive integer")
        for case in report["cases"]:
            if (not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"].strip()
                    or not isinstance(case.get("case_sha256"), str) or len(case["case_sha256"]) != 64
                    or case.get("expect") not in ("prediction", "review") or not isinstance(case.get("outcomes"), list)
                    or len(case["outcomes"]) != repeats or not all(isinstance(value, dict) for value in case["outcomes"])):
                raise ValueError("measurement must retain every repeated outcome")
            if any(outcome.get("manifest_sha256") != report["bundle_manifest_sha256"] for outcome in case["outcomes"]):
                raise ValueError("measurement outcomes belong to a different bundle")
    for field in ("bundle_manifest_sha256", "probe_sha256", "implementation_sha256", "runtime"):
        if left.get(field) != right.get(field):
            raise ValueError(f"measurements use different {field}")
    if ([case["id"] for case in left["cases"]] != [case["id"] for case in right["cases"]]
            or len({case["id"] for case in left["cases"]}) != len(left["cases"])):
        raise ValueError("measurement case IDs differ or repeat")
    mismatches, maximum = [], 0.0
    for a, b in zip(left["cases"], right["cases"]):
        if a["case_sha256"] != b["case_sha256"] or not a.get("outcomes") or not b.get("outcomes"):
            raise ValueError("measurement case inputs differ or outcomes are missing")
        reference = a["outcomes"][0]
        case_mismatch = False
        for actual in [*a["outcomes"], *b["outcomes"]]:
            expected_prediction = reference.get("prediction", reference.get("suggestion", {}))
            actual_prediction = actual.get("prediction", actual.get("suggestion", {}))
            fields = ("status", "reason", "question_id", "value", "manifest_sha256")
            if (any(type(actual.get(field)) is not type(reference.get(field)) or actual.get(field) != reference.get(field)
                    for field in fields)
                    or any(type(actual_prediction.get(field)) is not type(expected_prediction.get(field))
                           or actual_prediction.get(field) != expected_prediction.get(field)
                           for field in ("type", "value", "threshold"))):
                case_mismatch = True
            probabilities = expected_prediction.get("probabilities", [])
            values = actual_prediction.get("probabilities", [])
            if len(probabilities) != len(values) or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
                    for value in [*probabilities, *values]):
                raise ValueError("invalid or incompatible probability vectors")
            if probabilities and (not math.isclose(sum(probabilities), 1, abs_tol=1e-8)
                                  or not math.isclose(sum(values), 1, abs_tol=1e-8)):
                raise ValueError("probability vectors must sum to one")
            difference = max((abs(x - y) for x, y in zip(probabilities, values)), default=0)
            maximum = max(maximum, difference)
            case_mismatch |= difference > probability_atol
        if case_mismatch:
            mismatches.append(a["id"])
    return {"schema_version": 1, "stage": "deployment_comparison", "passed": not mismatches,
            "bundle_manifest_sha256": left["bundle_manifest_sha256"], "probe_sha256": left["probe_sha256"],
            "left_report_sha256": fingerprint(left), "right_report_sha256": fingerprint(right),
            "probability_atol": probability_atol, "max_probability_difference": maximum,
            "mismatched_cases": mismatches,
            "scope": "observed probes only; no accuracy, OOD or distribution-shift guarantee"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    for name in ("bundle", "expected_manifest_sha256", "probe", "output", "context"):
        run.add_argument("--" + name.replace("_", "-"), required=True)
    run.add_argument("--threads", type=int, required=True)
    comparison = commands.add_parser("compare")
    for name in ("left", "right", "output"):
        comparison.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if args.threads < 1:
            parser.error("--threads must be positive and match the bundle's recorded runtime")
        torch.set_num_threads(args.threads)
        probe = json.loads(Path(args.probe).read_text(encoding="utf-8"))
        result = measure(args.bundle, args.expected_manifest_sha256, probe, args.output, context=args.context)
        print(json.dumps({"output": args.output, "cases": len(result["cases"]), "verified_load_ms": result["verified_load_ms"]}))
    else:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError("comparison output already exists")
        result = compare(*(json.loads(Path(path).read_text(encoding="utf-8")) for path in (args.left, args.right)))
        output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_save(result, output)
        print(json.dumps(result))
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
