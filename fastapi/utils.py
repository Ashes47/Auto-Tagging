import base64
from PIL import Image
from constants import temp_file
import os
import urllib.request

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
    