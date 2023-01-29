from fastapi import FastAPI, Form
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
import threading
from models import Data
from utils import save_image_from_url
from face_recog import save_data, get_device, add_faces, recog_faces, load_data
from object_detection import get_tags_and_person_mask
from constants import CUSTOM_CLASS_LIST
from custom_object_detection import get_custom_tags
from custom_object_detection_utils import add_classes, train_custom_object_detection, clear_custom_class
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
    save_image_from_url(request.image)

    training_response = "No training requested"
    error_predicting = ""
    error_adding_data = ""
    response = {}
    tags_f = {
        "tag": [],
        "bbox": []
    }
    try:
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
    
    except Exception as e:
        error_predicting = e

    if request.tags and request.tags.get("tag"):
        tags = request.tags
        try:
            if tags["tag"].get("class") and tags["tag"].get("pixel_box"):
                face_addition = threading.Thread(target=add_faces, name="Add Face data", args=[tags["tag"]["class"], tags["tag"]["pixel_box"]])
                face_addition.start()

                class_addition = threading.Thread(target=add_classes, name="Add Custom Class", args=[tags["tag"]["class"], tags["tag"]["pixel_box"]])
                class_addition.start()
                training_response = "Added data for training"
            else:
                training_response = "Class or pixel box missing which is required for training"
        except Exception as e:
            error_adding_data = e


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
    response["error_predicting"] = error_predicting
    response["error_adding_data"] = error_adding_data
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
            CUSTOM_CLASS_LIST["training_status"] = False
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


@app.get("/set_training_status_false")
def set_training_status_false():
    CUSTOM_CLASS_LIST["training_status"] = False
    return {"result": "Succesfully Done"}


@app.post("/delete_custom_class")
def delete_custom_class(customClass: str = Form(default=None)):
    clear_custom_class(customClass)
    return {'result': 'Data reset succesfully'}
######################################################################################


############## Face Recognition ######################################################
@app.post("/delete_face")
def delete_face(person: str = Form(default=None)):
    embeddings, identity = load_data()
    while person in identity:
        print(f"Removing {person} from saved data")
        index = identity.index(person)
        identity = identity[:index] + identity[index+1:]
        embeddings = embeddings[:index] + embeddings[index+1:]
        save_data(embeddings, identity)
    return {'result': 'Data reset succesfully'}


@app.get("/get_faces_stored")
def get_faces_stored():
    _, identity = load_data()
    response = {"Faces Stored":identity}
    return response
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