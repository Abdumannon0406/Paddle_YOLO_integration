from ultralytics import YOLO

model=YOLO('/home/codeschool/Abdumannon/Historical_data_Recognition/detect/train15/weights/best.pt')

results= model.predict("image copy 4.png",save=True)