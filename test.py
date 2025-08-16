import cv2
import numpy as np
from pathlib import Path
from paddleocr import DocImgOrientationClassification,TextImageUnwarping
# Paths
input_folder = Path("Images")
output_folder = Path("rotated_images")
output_folder.mkdir(exist_ok=True)

model_or = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")
model_rec = TextImageUnwarping(model_name="UVDoc")


for img_path in input_folder.glob("*.*"):
    image = cv2.imread(str(img_path))
    if image is None:
        continue
    
    output = model_or.predict(image, batch_size=1)
    angle = 360-int(output[0]["label_names"][0])
    # Get dimensions
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation with warpAffine
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Save rotated image
    cv2.imwrite(str(output_folder / img_path.name), rotated)

print("Rotation complete!")
