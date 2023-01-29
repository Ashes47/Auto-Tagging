import os
import pickle

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

temp_file = "temp.jpg"
model_name = "models/yolov8m.pt"
custom_model_name = "models/custom_model.pt"
embeddings_name = "embeddings.txt"
identity_name = "identity.txt"

CUSTOM_CLASS_LIST = {
    "next_class_counter": 0,
    "split": [
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "valid",
        "valid",
        "valid"
        ],
    "next_unique_name": 0,
    "training_status": False,
    "class_list": []
}
if os.path.exists('CUSTOM_CLASS_LIST.txt'):
      with open('CUSTOM_CLASS_LIST.txt',"rb") as f:
          CUSTOM_CLASS_LIST = pickle.load(f)