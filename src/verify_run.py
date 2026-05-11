import sys
import os
import json
import subprocess
from pathlib import Path

def run_verification():
    print("\n" + "="*50)
    print("[START] Running MLOps Verification Suite...")
    print("="*50 + "\n")
    
    root_dir = Path(__file__).parent.parent
    results_file = root_dir / "results" / "training_summary.json"
    model_file = root_dir / "models" / "best_model.pth"
    log_dir = root_dir / "results" / "logs"
    
    errors = []
    
    # 1. Check Artifacts
    print("Checking artifacts...")
    if not results_file.exists():
        errors.append(f"Missing results file: {results_file.name}")
    if not model_file.exists():
        errors.append(f"Missing model weights: {model_file.name}")
    if not log_dir.exists() or len(list(log_dir.glob("events.out.tfevents.*"))) == 0:
        errors.append(f"Missing TensorBoard logs in {log_dir.name}")
        
    if errors:
        print("[FAIL] Artifact check failed!")
        for e in errors: print(f"  - {e}")
        return False
    print("[PASS] All required artifacts present.")
    
    # 2. Check Metrics Threshold
    print("Checking model performance metrics...")
    try:
        with open(results_file, "r") as f:
            summary = json.load(f)
        
        best_auc = summary.get("final_auc", summary.get("best_val_auc", 0.0))
        if best_auc < 0.80:
            errors.append(f"Model AUC ({best_auc}) is below the minimum threshold of 0.80")
            
    except Exception as e:
        errors.append(f"Error reading metrics: {e}")
        
    if errors:
        print("[FAIL] Metric threshold check failed!")
        for e in errors: print(f"  - {e}")
        return False
    print(f"[PASS] Model AUC ({best_auc}) meets the minimum threshold.")
    
    # 3. Linting Check
    print("Running code linting (flake8)...")
    try:
        # Ignore strict formatting for now (line lengths, etc) to focus on major issues
        result = subprocess.run(
            ["flake8", str(root_dir / "src"), "--ignore=E501,E402,E731,F401,W503,E702"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            errors.append(f"Flake8 linting failed:\n{result.stdout}")
    except Exception as e:
        errors.append(f"Flake8 execution failed: {e}")

    if errors:
        print("[FAIL] Code quality check failed!")
        for e in errors: print(f"  - {e}")
        return False
    print("[PASS] Code linting passed.")
    
    print("\n" + "="*50)
    print("[PASS] VERIFICATION SUITE PASSED. Safe to push.")
    print("="*50 + "\n")
    return True

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
