"""
PIPELINE.PY

Purpose:
Runs the Crop Recommendation project pipeline end-to-end.

Steps:
1. Clean raw dataset
2. Create engineered features and ML-ready dataset
3. Train machine learning models and save best model
4. Test prediction using saved model
"""

import subprocess
import sys


def run_step(step_name, command):
    print("\n" + "=" * 60)
    print(f"STARTING: {step_name}")
    print("=" * 60)

    subprocess.run(command, shell=True, check=True)

    print(f"COMPLETED: {step_name}")


def run_pipeline():
    print("\nCROP RECOMMENDATION PIPELINE STARTED")

    run_step(
        "Data Cleaning",
        f"{sys.executable} src/clean.py"
    )

    run_step(
        "Feature Engineering",
        f"{sys.executable} src/features.py"
    )

    run_step(
        "Model Training",
        f"{sys.executable} src/train.py"
    )

    run_step(
        "Prediction Test",
        f"{sys.executable} src/predict.py"
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()