from pydantic import BaseModel

class Data(BaseModel):
    image: str
    tags: dict

    class Config:
        schema_extra = {
            "example": {
                "image": "Image in b64 string format",
                "tags": {
                    "name": "Cristiano Ronaldo",
                    "tag": {
                        "class": "fish",
                        "pixel_box": [147, 54, 1073, 1200]
                    }
                }
            }
        }