import base64
from constants import temp_file
import os, shutil
import urllib.request
from PIL import Image


def imageToBase64(image):
    image.save(temp_file)
    with open(temp_file, "rb") as image_file:
      b64_image = str(base64.b64encode(image_file.read()))[1:-1]
    return b64_image


def save_image_from_url(url):
    urllib.request.urlretrieve(url, temp_file)
    Image.open(temp_file).convert("RGB").save(temp_file)


def base64ToImage(data):
    imgdata = base64.b64decode(data)
    with open(temp_file, 'wb') as f:
        f.write(imgdata)
    return


def clean_directory(path):
    shutil.rmtree(path)
    os.mkdir(path)