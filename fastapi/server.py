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


app = FastAPI(
    title="Auto-Tagging",
    description="""Auto Tag images""",
    version="1.0.0",
)


############## Auto Tagging ##########################################################
@app.post("/auto_tag")
async def auto_tagging(request: Data):
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
            print(await add_face(tags["name"]))
    
    if tags.get("tag"):
        if tags["tag"].get("class") and tags["tag"].get("pixel_box"):
            training_response = "Inititaing Training"
            await add_class(tags["tag"]["class"], tags["tag"]["pixel_box"])
        else:
            training_response = "Class or pixel box missing which is required for training"
    response["tags"] = set(generated_tags)
    response["training_response"] = training_response
    await clear_temp()
    return response
######################################################################################


############## Custom Object Detection ###############################################
@app.get("/train_custom_model")
async def train_custom_model():
    if len(CUSTOM_CLASS_LIST["class_list"]) == 0:
        return {'result': 'No data to train on'}
    if CUSTOM_CLASS_LIST["training_status"]:
        return {'result': 'Model already training'}
    else:
        CUSTOM_CLASS_LIST["training_status"] = True
        await train_custom_object_detection(25)
        return {'result': 'Training started'}


@app.get("/get_training_status")
async def get_training_status():
    if CUSTOM_CLASS_LIST["training_status"]:
        return {'result': 'Model training'}
    else:
        return {'result': 'Model trained'}


@app.get("/get_custom_class_info")
async def get_custom_class_info():
    return CUSTOM_CLASS_LIST
######################################################################################


############## Face Recognition ######################################################
@app.get("/reset_facial_data")
async def reset_facial_data():
    save_data([], [])
    return {'result': 'Face data reset succesfully'}
######################################################################################


############## Threads ##############################################################
@app.get("/threads")
async def get_threads_running():
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