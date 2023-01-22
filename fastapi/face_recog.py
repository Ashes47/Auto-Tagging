import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
from scipy.spatial.distance import cosine
from PIL import Image
import cv2
import os
from constants import temp_file, embeddings_name, identity_name
from object_detection import show_crop


mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, keep_all=True
)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def load_data():
    embeddings = []
    identity = []
    if os.path.exists(embeddings_name):
      with open(embeddings_name,"rb") as f:
          embeddings = pickle.load(f)
    if os.path.exists(identity_name):
      with open(identity_name,"rb") as f:
          identity = pickle.load(f)
    return embeddings, identity


def save_data(embeddings, identity):
  with open(embeddings_name, "wb") as fp: 
    pickle.dump(embeddings, fp)    
  with open(identity_name, "wb") as fp:  
    pickle.dump(identity, fp)


def get_accurate_detections(faces, probs):
  aligned = []
  for face, prob in zip(faces, probs):
    print(prob)
    if prob > 0.6:
      if face is not None:
        aligned.append(face.squeeze(0))
  if aligned:
    return torch.stack(aligned)
  else:
    return torch.tensor([])
  

def get_emb(image):
  boxes, probs = mtcnn.detect(Image.fromarray(image))
  faces = mtcnn(image)

  if probs[0] == None:
    print("No face found")
    return [], []

  pixelbox = []
  for i, k in zip(boxes, probs):
    if k > 0.6:
      temp = []
      for j in i:
        temp.append(int(j))
      pixelbox.append(temp)
  
  # images = []
  # for box in pixelbox:
  #   images.append(show_crop(image, box))

  # print(f"Cropped Faces found {len(images)}")

  aligned = get_accurate_detections(faces, probs)
  if len(aligned) == 0:
    return [], []
  embeddings = resnet(aligned).detach().cpu()
  return embeddings, pixelbox


def add_face(name, image):
  embeddings, identity = load_data()
  emb, pixelbox = get_emb(image)
  if len(emb) == 0:
    return "No face found"
  embeddings.append(emb[0])
  identity.append(name)
  save_data(embeddings, identity)
  return "Face successfully added"


def match_face_with_database(new_emb):
  embeddings, identity = load_data()
  if len(embeddings) == 0:
    return ""
  ans = []
  for emb in embeddings:
      ans.append(cosine(emb,new_emb))
  index = np.argsort(ans)
  per = (1-ans[index[0]])*100
  prettyPer = "{:.2f}".format(per)
  print('Matched with {} with {}%'.format(identity[index[0]],prettyPer))
  if per > 60.00:
    return identity[index[0]]
  return ""


def recog_faces():
  image = cv2.imread(temp_file)
  embs, face_bbox  = get_emb(image)
  print(face_bbox)
  tags = []
  bbox = []
  for i, emb in enumerate(embs):
    name = match_face_with_database(emb)
    print(name)
    print(i)
    print(face_bbox[i])
    if name != "":
      tags.append(name)
      bbox.append(face_bbox[i])
  return tags, bbox

def add_faces(classes, pixelboxes):
  print(f"{classes} - {pixelboxes}")
  for new_class, pixelbox in zip(classes, pixelboxes):
    print(f"{new_class} is being added")
    image = cv2.imread(temp_file)
    image = show_crop(image, pixelbox)
    print(add_face(new_class, image))