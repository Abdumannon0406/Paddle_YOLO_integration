from ultralytics import YOLO
from pathlib import Path
import cv2

model=YOLO("/home/codeschool/Abdumannon/Historical_data_Recognition/detect/train15/weights/best.pt")

images_folder=Path("Images")

images_paths=list(images_folder.glob('*.*'))

for i in range(len(images_paths)):
    image=cv2.imread(images_paths[i])
    image_stem=Path(images_paths[i]).stem
    results=model.predict(images_paths[i])

    keypoints=results[0].boxes.xyxy.cpu().numpy().astype(int)
    classes=results[0].boxes.cls.cpu().numpy()
    for j in range(len(keypoints)):
        if int(classes[j])==2:
            cropped_image=image[keypoints[j][1]:keypoints[j][3],keypoints[j][0]:keypoints[j][2]]
            cv2.imwrite(f'Output_images/{image_stem}_{j}.jpg',cropped_image)


