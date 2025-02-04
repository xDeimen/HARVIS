from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("yolo11n.pt")

    # Train model
    model.train(data="CarEngineBayDataset/data.yaml", epochs=50, imgsz=640, batch=16)
    model.val()


    model.export(format="onnx")  # Convert to ONNX