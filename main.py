from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from function import main_process
import validators
import uvicorn

app = FastAPI()

@app.get("/")
async def rootMsg():
    return "API IS RUNNING PERFECTLY"

class ImageRequest(BaseModel):
    floorImage: str
    tileImage: str

    # Custom validator for image URLs
    @validator('floorImage', 'tileImage')
    def url_must_be_valid(cls, v):
        if not validators.url(v):
            raise ValueError('Please provide a valid URL')
        return v

@app.post("/floormasking")
async def perform_image_processing(request: ImageRequest):
    try:
        # If we're here, the URLs are valid
        uploaded_image_url = main_process(request.floorImage, request.tileImage)
        return {"uploaded_image_url": uploaded_image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
