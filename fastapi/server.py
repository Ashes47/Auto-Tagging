from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
import threading
from models import Data
from utils import base64ToImage, clear_temp
from face_recog import save_data, get_device, add_face, recog_faces
from object_detection import get_tags_and_person_mask
from constants import CUSTOM_CLASS_LIST
from custom_object_detection import get_custom_tags
from custom_object_detection_utils import add_class, train_custom_object_detection
import os


app = FastAPI(
    title="Auto-Tagging",
    description="""Auto Tag images""",
    version="1.0.0",
)


############## Auto Tagging ##########################################################
@app.post("/auto_tag")
def auto_tagging(request: Data):
    base64ToImage(request.image)
    tags = request.tags
    training_response = "No training requested"
    response = {}

    persons, generated_tags = get_tags_and_person_mask()
    custom_tags = get_custom_tags()

    for custom_tag in custom_tags:
        generated_tags.append(custom_tag)

    if "person" in generated_tags:
        for person in persons:
            name = recog_faces(person)
            if name != []:
                generated_tags.append(name[0])
        if tags.get("name"):
            face_addition = threading.Thread(target=add_face, name="Add Face data", args=[tags["name"]])
            face_addition.start()
    
    if tags.get("tag"):
        if tags["tag"].get("class") and tags["tag"].get("pixel_box"):
            training_response = "Added data for training"
            class_addition = threading.Thread(target=add_class, name="Add Custom Class", args=[tags["tag"]["class"], tags["tag"]["pixel_box"]])
            class_addition.start()
        else:
            training_response = "Class or pixel box missing which is required for training"
    response["tags"] = set(generated_tags)
    response["training_response"] = training_response
    temp_clear = threading.Thread(target=clear_temp, name="Clear Temp files")
    temp_clear.start()
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
@app.get("/reset_facial_data")
def reset_facial_data():
    save_data([], [])
    return {'result': 'Face data reset succesfully'}
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