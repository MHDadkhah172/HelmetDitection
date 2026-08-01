from ultralytics import YOLO

def train_model(data_yaml="data/data.yaml", epochs=15, batch_size=16, img_size=640):
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name="helmet_yolo11_run",
        project="runs/detect",
        save=True,
    )

if __name__ == "__main__":
    yaml_path = "data/data.yaml"
    train_model(data_yaml=yaml_path, epochs=15, batch_size=16)