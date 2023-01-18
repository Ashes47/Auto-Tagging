from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
import threading
from models import Data
from utils import base64ToImage, clear_temp
from face_recog import save_data, get_device, add_face, recog_faces
from object_detection import get_tags_and_person_mask

app = FastAPI(
    title="Auto-Tagging",
    description="""Auto Tag images""",
    version="1.0.0",
)

############## THREADS ##############################################################
@app.get("/threads")
async def get_threads_running():
    return {
        "threads_running": threading.active_count(),
        "device": get_device()
        }
######################################################################################

############## Auto Tagging ##########################################################
@app.post("/auto_tag")
async def auto_tagging(request: Data):
    base64ToImage(request.image)
    tags = request.tags

    persons, generated_tags = get_tags_and_person_mask()
    if tags.get("category") == "human":
        for person in persons:
            name = recog_faces(person)
            if name != []:
                generated_tags.append(name[0])
        if tags.get("name"):
            print(await add_face(tags["name"]))
    response = {"tags": set(generated_tags)}
    await clear_temp()
    return response
######################################################################################

############## Face Recognition ######################################################
@app.get("/reset_facial_data")
async def reset_facial_data():
    save_data([], [])
    return "{'result': 'Face data reset succesfully'}"
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