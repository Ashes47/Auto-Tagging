from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
import threading
import numpy as np
from models import Data
from utils import save_image_from_url, clear_temp,clear_custom_data
from face_recog import save_data, get_device, add_faces, recog_faces
from object_detection import get_tags_and_person_mask
from constants import CUSTOM_CLASS_LIST
from custom_object_detection import get_custom_tags
from custom_object_detection_utils import add_classes, train_custom_object_detection
import os


app = FastAPI(
    title="Auto-Tagging",
    description="""Auto Tag images""",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

############## Auto Tagging ##########################################################
@app.post("/auto_tag")
def auto_tagging(request: Data):
    clear_temp()
    save_image_from_url(request.image)

    training_response = "No training requested"
    response = {}
    tags_f = {
        "tag": [],
        "bbox": []
    }
    generated_tags, bounding_box_object = get_tags_and_person_mask()

    for generated_tag, bbox, in zip(generated_tags, bounding_box_object):
        tags_f["tag"].append(generated_tag)
        tags_f["bbox"].append(bbox)

    custom_tags , bounding_box_custom= get_custom_tags()

    for custom_tag, bounding_box_custome_one in zip(custom_tags, bounding_box_custom):
        tags_f["tag"].append(custom_tag)
        tags_f["bbox"].append(bounding_box_custome_one)

    name, bounding_box_face = recog_faces()
    for generated_tag, bbox, in zip(name, bounding_box_face):
        tags_f["tag"].append(generated_tag)
        tags_f["bbox"].append(bbox)

    if request.tags and request.tags.get("tag"):
        tags = request.tags
        if tags["tag"].get("class") and tags["tag"].get("pixel_box"):
            face_addition = threading.Thread(target=add_faces, name="Add Face data", args=[tags["tag"]["class"], tags["tag"]["pixel_box"]])
            face_addition.start()

            training_response = "Added data for training"
            class_addition = threading.Thread(target=add_classes, name="Add Custom Class", args=[tags["tag"]["class"], tags["tag"]["pixel_box"]])
            class_addition.start()
        else:
            training_response = "Class or pixel box missing which is required for training"

    response_tags = []
    print(tags_f)
    for i in tags_f["bbox"]:
        temp = []
        for j in i:
            j = temp.append(str(j))
        response_tags.append(temp)
    tags_f["bbox"] = response_tags

    response["tags"] = tags_f
    response["training_response"] = training_response
    print(response)
    return response
######################################################################################


############## Custom Object Detection ###############################################
@app.get("/train_custom_model")
def train_custom_model():
    if len(CUSTOM_CLASS_LIST["class_list"]) == 0:
        return {'result': 'No data to train on'}

    if CUSTOM_CLASS_LIST["training_status"]:
        return {'result': 'Model already training'}
    else:
        try:
            CUSTOM_CLASS_LIST["training_status"] = True
            train_custom_model = threading.Thread(target=train_custom_object_detection, name="Train custom model", args=[25])
            train_custom_model.start()
        except Exception as e:
            return {'result': 'Training Failed', 'error': e}
        return {'result': 'Training started'}


@app.get("/get_training_status")
def get_training_status():
    if CUSTOM_CLASS_LIST["training_status"]:
        return {'result': 'Model is training...'}
    else:
        if os.path.exists("./models/custom_model.pt"):
            return {'result': 'Model trained'}
        else:
            return {'result': 'No model trained'}


@app.get("/get_custom_class_info")
def get_custom_class_info():
    return CUSTOM_CLASS_LIST
######################################################################################


############## Face Recognition ######################################################
@app.get("/reset_training_data")
def reset_training_data():
    save_data([], [])
    clear_custom_data()    
    return {'result': 'Data reset succesfully'}
######################################################################################


############## Threads ##############################################################
@app.get("/threads")
def get_threads_running():
    return {
        "threads_running": threading.active_count(),
        "device": get_device()
        }
######################################################################################


############## Documentation #########################################################
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="API Documentation",
        version="1.0.0",
        description="Please follow below to use this service",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
######################################################################################