from pathlib import Path
from ultralytics import YOLO

def evaluate_model():
    project_dir = Path(__file__).resolve().parent.parent
    model_path = project_dir / "models" / "best.pt"
    yaml_path = project_dir / "data" / "data.yaml"

    model = YOLO(str(model_path))
    metrics = model.val(data=str(yaml_path), split="test")


    print("--- Final Model Evaluation Results (Step 5) ---")

    print(f"Precision:      {metrics.results_dict['metrics/precision(B)']:.2f}")
    print(f"Recall:         {metrics.results_dict['metrics/recall(B)']:.2f}")
    print(f"mAP50:          {metrics.results_dict['metrics/mAP50(B)']:.2f}")
    print(f"mAP50-95:       {metrics.results_dict['metrics/mAP50-95(B)']:.2f}")

if __name__ == "__main__":
    evaluate_model()
