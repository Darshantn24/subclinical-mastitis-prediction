from src.training.train import run_training
import json
from pathlib import Path



if __name__ == "__main__":
    model, results, model_path = run_training()

    print("\nMODEL EVALUATION RESULTS")
    print("-" * 35)

    for metric, value in results.items():
        print(f"{metric}: {value}")

    print(f"\nModel saved at: {model_path}")

report_dir = Path("reports")
report_dir.mkdir(exist_ok=True)

with open(report_dir / "evaluation_report.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nEvaluation report saved successfully.")