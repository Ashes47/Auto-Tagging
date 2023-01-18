import base64
from PIL import Image
import os

def imageToBase64(image):
    image.save('temp.jpeg')
    with open('temp.jpeg', "rb") as image_file:
      b64_image = str(base64.b64encode(image_file.read()))[1:-1]
    os.remove('temp.jpeg')
    return b64_image

def base64ToImage(data):
    imgdata = base64.b64decode(data)
    with open('temp.jpeg', 'wb') as f:
        f.write(imgdata)
    image = Image.open('temp.jpeg')
    return image

async def clear_temp():
    os.remove('temp.jpeg')
    