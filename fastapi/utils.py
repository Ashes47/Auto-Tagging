import base64
from constants import temp_file
import os, shutil
import urllib.request
import pickle


def imageToBase64(image):
    image.save(temp_file)
    with open(temp_file, "rb") as image_file:
      b64_image = str(base64.b64encode(image_file.read()))[1:-1]
    return b64_image


def save_image_from_url(url):
    urllib.request.urlretrieve(url, temp_file)


def base64ToImage(data):
    imgdata = base64.b64decode(data)
    with open(temp_file, 'wb') as f:
        f.write(imgdata)
    return


def clear_temp():
    if os.path.exists(temp_file):
        os.remove(temp_file)


def clean_directory(path):
    shutil.rmtree(path)
    os.mkdir(path)


def clear_custom_data():
    if os.path.exists("./models/custom_model.pt"):
        os.remove("./models/custom_model.pt")
    if os.path.exists('CUSTOM_CLASS_LIST.txt'):
        os.remove('CUSTOM_CLASS_LIST.txt')

    clean_directory('./custom_dataset/train/images')
    clean_directory('./custom_dataset/train/labels')
    clean_directory('./custom_dataset/valid/images')
    clean_directory('./custom_dataset/valid/labels')

    if os.path.exists('./custom_dataset/train/labels.cache'):
        os.remove('./custom_dataset/train/labels.cache')

    if os.path.exists('./custom_dataset/valid/labels.cache'):
        os.remove('./custom_dataset/valid/labels.cache')

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
        "next_unique_name": 1,
        "training_status": False,
        "class_list": []
    }

    with open('CUSTOM_CLASS_LIST.txt', "wb") as fp:  
        pickle.dump(CUSTOM_CLASS_LIST, fp)