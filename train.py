from ultralytics import YOLO

model=YOLO("yolo11n.pt")

results=model.train(data="Dataset/data.yaml",epochs=100)