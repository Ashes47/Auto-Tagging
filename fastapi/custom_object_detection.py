from ultralytics import YOLO
from constants import CUSTOM_CLASS_LIST, custom_model_name, temp_file
import os

if os.path.exists(custom_model_name):
    custom_model = YOLO(custom_model_name)
    custom_model.fuse()
    
def get_custom_tags():
    tags = []
    bbox = []
    if not os.path.exists(custom_model_name):
        return tags
    results = custom_model.predict(source=temp_file, conf=0.7)
    for result in results:
        boxes = result.cpu().boxes.numpy()
    for i in range(0,len(boxes)):
        bbox.append(boxes.xyxy[i])
        print(f"{CUSTOM_CLASS_LIST['class_list'][int(boxes.cls[i])]}: {boxes.conf[i]}%")
        tags.append(CUSTOM_CLASS_LIST["class_list"][int(boxes.cls[i])])
    return tags, bbox
