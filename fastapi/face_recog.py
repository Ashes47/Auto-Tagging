import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
from scipy.spatial.distance import cosine
from PIL import Image
import cv2
import os
from constants import temp_file, embeddings_name, identity_name, pixelboxes_name
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
    pixelboxes =[]
    if os.path.exists(embeddings_name):
      with open(embeddings_name,"rb") as f:
          embeddings = pickle.load(f)
    if os.path.exists(identity_name):
      with open(identity_name,"rb") as f:
          identity = pickle.load(f)
    if os.path.exists(pixelboxes_name):
      with open(pixelboxes_name,"rb") as f:
          pixelboxes = pickle.load(f)
    return embeddings, identity, pixelboxes


def save_data(embeddings, identity, pixelboxes):
  with open(embeddings_name, "wb") as fp: 
    pickle.dump(embeddings, fp)    
  with open(identity_name, "wb") as fp:  
    pickle.dump(identity, fp)
  with open(pixelboxes_name, "wb") as fp:  
    pickle.dump(pixelboxes, fp)


def get_accurate_detections(aligned_images, probs):
  aligned = []
  for image, prob in zip(aligned_images, probs):
    if prob > 0.9:
      aligned.append(image)
  if aligned:
    return torch.stack(aligned)
  else:
    return torch.tensor([])
  

def get_emb(image):
  aligned_images, probs = mtcnn(image, return_prob=True)
  print(probs)
  if probs[0] == None:
    print("No face found")
    return []
  aligned = get_accurate_detections(aligned_images, probs)
  if len(aligned) == 0:
    return []
  embeddings = resnet(aligned).detach().cpu()
  return embeddings


def add_face(name, image, bboxes):
  embeddings, identity, pixelboxes = load_data()
  emb = get_emb(image)
  if len(emb) == 0:
    return "No face found"
  embeddings.append(emb[0])
  identity.append(name)

  bboxes_save = []
  for i in bboxes:
    temp = []
    for j in i:
      j = temp.append(str(j))
    bboxes_save.append(temp)
  pixelboxes.append(bboxes_save)
  
  save_data(embeddings, identity, pixelboxes)
  return "Face successfully added"


def match_face_with_database(new_emb):
  embeddings, identity, pixelboxes = load_data()
  if len(embeddings) == 0:
    return [], []
  ans = []
  for emb in embeddings:
      ans.append(cosine(emb,new_emb))
  index = np.argsort(ans)
  per = (1-ans[index[0]])*100
  prettyPer = "{:.2f}".format(per)
  print('Matched with {} with {}%'.format(identity[index[0]],prettyPer))
  if per > 70.00:
    return identity[index[0]], pixelboxes[index[0]]
  return [], []


def recog_faces(image):
  embs  = get_emb(image)
  tags = []
  bbox = []
  for emb in embs:
    name, box = match_face_with_database(emb)
    if name != []:
      tags.append(name)
      bbox.append(box)
  return tags, bbox

def add_faces(classes, pixelboxes):
  for new_class, pixelbox in zip(classes, pixelboxes):
    print(f"{new_class} is being added")
    image = cv2.imread(temp_file)
    image = show_crop(image, pixelbox)
    print(add_face(new_class, Image.fromarray(image), pixelboxes))