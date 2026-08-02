import os
from ultralytics import YOLO


def run_inference(source_path, model_path="../models/best.pt", conf_threshold=0.35):

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return

    if not os.path.exists(source_path):
        print(f"Error: Source path not found at '{source_path}'")
        return

    model = YOLO(model_path)


    results = model.predict(
        source=source_path,
        conf=conf_threshold,
        save=True,
        project="../docs",
        name="predictions",
        exist_ok=True
    )

    for idx, result in enumerate(results):
        boxes = result.boxes
        names = result.names

        helmet_count = sum(1 for box in boxes if names[int(box.cls[0])] == 'helmet')
        head_count = sum(1 for box in boxes if names[int(box.cls[0])] == 'head')

        print(f"\n--- Image [{idx + 1}/{len(results)}]: {os.path.basename(result.path)} ---")
        print(f"Helmets (Safe):   {helmet_count}")
        print(f"Heads (Unsafe):   {head_count}")

        total = helmet_count + head_count
        if total > 0:
            print(f"Compliance Rate: {(helmet_count / total) * 100:.1f}%")


if __name__ == "__main__":
    TEST_SOURCE = "../data/test_images"
    run_inference(source_path=TEST_SOURCE, conf_threshold=0.35)