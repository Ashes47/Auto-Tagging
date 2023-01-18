from pydantic import BaseModel

class Data(BaseModel):
    image: str
    tags: dict

    class Config:
        schema_extra = {
            "example": {
                "image": "Image in b64 string format",
                "tags": {
                    "category":"human",
                    "name": "Cristiano Ronaldo"
                }
            }
        }