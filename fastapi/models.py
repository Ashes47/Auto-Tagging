from pydantic import BaseModel

class Data(BaseModel):
    image: str
    tags: dict