import cv2
from ultralytics import YOLO
from constants import CLASS_LIST, model_name, temp_file

model = YOLO(model_name)
model.fuse()

def show_crop(img, box):
  test_arr = []
  img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
  test_arr.append(img)
  return test_arr

def get_tags_and_person_mask():
  results = model.predict(source=temp_file, conf=0.7)
  persons = []
  tags = []
  for result in results:
    boxes = result.cpu().boxes.numpy()
    for i in range(0,len(boxes)):
      print(f"{CLASS_LIST[int(boxes.cls[i])]}: {boxes.conf[i]}%")
      if CLASS_LIST[int(boxes.cls[i])] == "person":
        persons.append(show_crop(cv2.imread(temp_file), boxes.xyxy[i]))
      tags.append(CLASS_LIST[int(boxes.cls[i])])
  return (sum(persons, [])), tags
