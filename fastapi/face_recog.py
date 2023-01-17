import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
from scipy.spatial.distance import cosine

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, keep_all=True
)

resnet = InceptionResnetV1(pretrained='vggface2').eval()

def load_data():
    embeddings=[]
    identity = []
    with open("embeddings.txt","rb") as f:
        embeddings = pickle.load(f)
    with open("identity.txt","rb") as f:
        identity = pickle.load(f)

    return embeddings, identity

def save_data(embeddings, identity):
  with open("embeddings.txt", "wb") as fp: 
    pickle.dump(embeddings, fp)    
  with open("identity.txt", "wb") as fp:  
      pickle.dump(identity, fp)

def get_accurate_detections(aligned_images, probs):
  aligned = []
  for image, prob in zip(aligned_images, probs):
    if prob > 0.9:
      aligned.append(image)
  return torch.stack(aligned)
  
def get_emb(image):
  aligned_images, probs = mtcnn(image, return_prob=True)
  if probs == [None]:
    print("No face found")
    return []
  aligned = get_accurate_detections(aligned_images, probs)
  embeddings = resnet(aligned).detach().cpu()
  return embeddings


async def add_face(image, name):
  embeddings, identity = load_data()
  emb = get_emb(image)
  if len(emb) == 0:
    return "No face found"
  embeddings.append(emb[0])
  identity.append(name)
  save_data(embeddings, identity)
  return "Face successfully added"

def match_face_with_database(new_emb):
  embeddings, identity = load_data()
  ans = []
  for emb in embeddings:
      ans.append(cosine(emb,new_emb))
  index = np.argsort(ans)
  per = (1-ans[index[0]])*100
  prettyPer = "{:.2f}".format(per)
  print('Matched with {} with {}%'.format(identity[index[0]],prettyPer))
  if per > 70.00:
    return identity[index[0]]
  return []

def recog_faces(image):
  embs = get_emb(image)
  tags = []
  for emb in embs:
    name = match_face_with_database(emb)
    if name != []:
      tags.append(name)
  return tags
