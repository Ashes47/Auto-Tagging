import cv2
from ultralytics import YOLO
from constants import CLASS_LIST, model_name, temp_file

model = YOLO(model_name)
model.fuse()

def show_crop(img, box):
    x1 = min(int(box[1]), int(box[3]))
    x2 = max(int(box[1]), int(box[3]))
    y1 = min(int(box[0]), int(box[2]))
    y2 = max(int(box[0]), int(box[2]))
    img = img[x1:x2, y1:y2]
    return img

def get_tags_and_person_mask():
  results = model.predict(source=temp_file, conf=0.7)
  tags = []
  bbox= []
  for result in results:
    boxes = result.cpu().boxes.numpy()
    for i in range(0,len(boxes)):
      bbox.append(boxes.xyxy[i])
      print(f"{CLASS_LIST[int(boxes.cls[i])]}: {boxes.conf[i]}%")
      tags.append(CLASS_LIST[int(boxes.cls[i])])
  print(bbox)
  return tags, bbox
