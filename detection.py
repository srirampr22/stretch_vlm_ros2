# usr/bin/env python3
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

model = load_model("/home/sriram/vlm_ws/src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/sriram/vlm_ws/src/GroundingDINO/weights/groundingdino_swint_ogc.pth")
IMAGE_PATH = "/home/sriram/vlm_ws/src/stretch_vlm_pkg/image_folder/study_desk.jpg"
TEXT_PROMPT = "phone ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imshow('Annotated Image', annotated_frame)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cv2.imwrite("annotated_image.jpg", annotated_frame)