from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator  # Importing validator here
from function import main_process  # Importing main_process function
import validators  # Import the validators module
import uvicorn
app = FastAPI()

@app.get("/")
async def rootMsg():
    return "API IS RUNNING PERFECTLY"

class ImageRequest(BaseModel):
    floorImage: str
    tileImage: str

    # Custom validator for image URLs
    @validator('floorImage', 'tileImage', each_item=True)  # Added each_item=True for individual validation
    def url_must_be_valid(cls, v):
        if not validators.url(v):
            raise ValueError('Please provide a valid URL')
        return v

@app.post("/floormasking")
async def perform_image_processing(request: ImageRequest):
    try:
        # If we're here, the URLs are valid
        uploaded_image_url = main_process(request.floorImage, request.tileImage)  # Using main_process function
        return {"uploaded_image_url": uploaded_image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5173, reload=True)