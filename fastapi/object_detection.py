import os
import cv2
from ultralytics import YOLO

CLASS_LIST = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

model = YOLO('yolov8s.pt')
model.fuse()

def show_crop(img, box):
  test_arr = []
  img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
  test_arr.append(img)
  return test_arr

def detect_object():
  results = model.predict(source='temp.jpg', conf=0.7)
  persons = []
  tags = []
  for result in results:
    boxes = result.cpu().boxes.numpy()
    for i in range(0,len(boxes)):
      print(f"{CLASS_LIST[int(boxes.cls[i])]}: {boxes.conf[i]}%")
      if CLASS_LIST[int(boxes.cls[i])] == "person":
        persons.append(show_crop(cv2.imread("temp.jpg"), boxes.xyxy[i]))
      tags.append(CLASS_LIST[int(boxes.cls[i])])
  os.remove('temp.jpg')
  return (sum(persons, [])), tags

def get_tags_and_person_mask(image):
  image.save('temp.jpg')
  return detect_object()