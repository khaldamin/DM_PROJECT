"""
Master Experiments Runner for Cupid AI.
Orchestrates and executes all enhancement experiment scripts sequentially,
monitoring execution times and compiling a comprehensive status report.
"""

import os
import sys
import time
import subprocess

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPTS = [
    ("Diagnostic Error Analysis & Calibration", "experiments/error_analysis.py"),
    ("Feature Engineering Ablation Study", "experiments/ablation_study.py"),
    ("Learning Curves & Sample Efficiency", "experiments/learning_curves.py"),
    ("Multi-Criteria Feature Selection", "experiments/feature_selection_analysis.py"),
    ("Model Robustness & Perturbation Test", "experiments/model_robustness.py"),
    ("Statistical Hypothesis Significance", "experiments/statistical_model_comparison.py"),
    ("Extended Clustering & Radar Profiling", "experiments/clustering_extended_analysis.py"),
    ("Advanced SHAP & Feature Interactions", "experiments/advanced_shap_analysis.py"),
]


def check_prerequisites():
    """Verify presence of data and core artifacts."""
    required_files = [
        os.path.join("data", "raw", "cupid_algorithm_dataset.csv"),
        os.path.join("results", "best_model.pkl"),
        os.path.join("results", "preprocessor.pkl"),
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("Warning: The following files are missing:")
        for m in missing:
            print(f"  - {m}")
        print("Run `python3 src/train_and_evaluate.py` first if artifacts are missing.\n")
        return False
    return True


def run_all():
    print("==================================================")
    print(" Cupid AI: Master Enhancement Experiments Suite")
    print("==================================================")
    start_total = time.time()

    check_prerequisites()

    results_summary = []

    for name, script_path in SCRIPTS:
        print(f"\n>>> Running: {name} ({script_path})")
        start_step = time.time()
        try:
            res = subprocess.run([sys.executable, script_path], check=True, text=True, capture_output=False)
            elapsed = time.time() - start_step
            status = "SUCCESS"
            print(f"[OK] Completed in {elapsed:.1f}s")
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_step
            status = "FAILED"
            print(f"[FAIL] Failed in {elapsed:.1f}s (Exit code: {e.returncode})")
        except Exception as ex:
            elapsed = time.time() - start_step
            status = "ERROR"
            print(f"[ERROR] Exception: {ex}")

        results_summary.append({
            "Experiment": name,
            "Script": script_path,
            "Status": status,
            "Duration_Seconds": round(elapsed, 1)
        })

    total_elapsed = time.time() - start_total
    print("\n==================================================")
    print(" Experiments Execution Summary")
    print("==================================================")
    for r in results_summary:
        symbol = "[OK]  " if r["Status"] == "SUCCESS" else "[FAIL]"
        print(f"{symbol} {r['Experiment']:40s} : {r['Status']} ({r['Duration_Seconds']}s)")
    print(f"\nTotal elapsed time: {total_elapsed / 60:.1f} minutes")
    print("==================================================")


if __name__ == '__main__':
    run_all()
