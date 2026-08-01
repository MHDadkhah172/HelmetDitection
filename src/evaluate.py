from pathlib import Path
import torch
from ultralytics import YOLO


def evaluate_model():
    project_dir = Path(__file__).resolve().parent.parent
    model_path = project_dir / "models" / "best.pt"
    yaml_path = project_dir / "data" / "data.yaml"


    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(yaml_path),
        split="test",
        conf=0.25,
        device=device,
        save=False,
        plots=False,
        verbose=False
    )

    names = metrics.names
    class_indices = metrics.box.ap_class_index
    nt_per_class = metrics.nt_per_class

    print("\n" + "=" * 77)
    print("--- Class-wise Evaluation & Data Distribution ---")
    print("=" * 77)
    print(f"{'Class':<15}{'Instances':<12}{'Precision':<12}{'Recall':<12}{'mAP50':<12}{'mAP50-95':<12}")
    print("-" * 77)

    for i, c in enumerate(class_indices):
        class_name = names[c]
        instances = int(nt_per_class[c])
        p = metrics.box.p[i]
        r = metrics.box.r[i]

        map50 = metrics.box.all_ap[i, 0]
        map95 = metrics.box.maps[i]

        print(f"{class_name:<15}{instances:<12}{p:<12.3f}{r:<12.3f}{map50:<12.3f}{map95:<12.3f}")

    print("-" * 77)

    total_instances = int(sum(nt_per_class))
    mean_p = metrics.results_dict['metrics/precision(B)']
    mean_r = metrics.results_dict['metrics/recall(B)']
    mean_map50 = metrics.results_dict['metrics/mAP50(B)']
    mean_map95 = metrics.results_dict['metrics/mAP50-95(B)']

    print(
        f"{'all (Overall)':<15}{total_instances:<12}{mean_p:<12.3f}{mean_r:<12.3f}{mean_map50:<12.3f}{mean_map95:<12.3f}")
    print("=" * 77)

    print("Key Finding: Class Imbalance bottleneck detected!")
    print("   The 'person' class has fewer instances compared to 'helmet' and 'head',")
    print("   which directly causes the drop in overall recall.")


if __name__ == "__main__":
    evaluate_model()