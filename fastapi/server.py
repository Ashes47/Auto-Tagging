from fastapi import FastAPI, File, Form, HTTPException
from fastapi.openapi.utils import get_openapi
import threading
import io
from PIL import Image
from face_recog import save_data, get_device, add_face, recog_faces
from object_detection import get_tags_and_person_mask

app = FastAPI(
    title="Upscale Images",
    description="""Upscale images using RealESRGAN""",
    version="0.1.0",
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
@app.post("/auto_tagging")
async def auto_tagging(image: bytes = File(default=None), tags: str = Form(default=None)):
    if image is None:
        raise HTTPException(status_code=404, detail="Data unavailable")
    input_image = Image.open(io.BytesIO(image)).convert("RGB")
    tag = {}
    if tags is not None:
        for i in tags.split(','):
            x = i.split(':')
            tag[x[0]] = x[1]
    persons, generated_tags = get_tags_and_person_mask(input_image)
    if tag.get("category") == "human":
        for person in persons:
            name = recog_faces(person)
            if name != []:
                generated_tags.append(name[0])
        if tag.get("name"):
            print(await add_face(input_image, tag["name"]))
    response = {"tags": set(generated_tags)}
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