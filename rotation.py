from paddleocr import DocImgOrientationClassification,TextImageUnwarping
from pathlib import Path
import cv2

# Load models
model_or = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")
model_rec = TextImageUnwarping(model_name="UVDoc")
# Read image
image_path = "rotation2.png"
image = cv2.imread(image_path)

# Predict orientation
output = model_or.predict(image_path, batch_size=1)
angle = 360-int(output[0]["label_names"][0])  # Get predicted angle

def rotate_image(image, angle):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

# Rotate
rotated = rotate_image(image, angle)

# Save rotated image
cv2.imwrite("rotated_image.png", rotated)
print(f"Rotated by {angle}° and saved to rotated_image.png")


outputs2=model_rec.predict(rotated,batch_size=1)

for res in outputs2:
    res.save_to_img(save_path="rectification/result.png")