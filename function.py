import requests
import cv2
import numpy as np
from PIL import Image
import base64
import io
import datetime
import os
import urllib.request
import firebase_admin
from firebase_admin import credentials, storage
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Firebase credentials
firebase_cred = credentials.Certificate({
    "type": os.getenv("FIREBASE_TYPE"),
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
    "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL")
})

firebase_admin.initialize_app(firebase_cred, {'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET")})

# Hugging Face credentials
API_URL = os.getenv("HF_API_URL")
headers = {"Authorization": f"Bearer {os.getenv('HF_AUTH_TOKEN')}"}

def query_api(filename):
    try:
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(API_URL, headers=headers, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")
        return None

def load_images(floor_filename, tile_filename):
    try:
        floor_image = cv2.imread(floor_filename)
        tile_image = cv2.imread(tile_filename, cv2.IMREAD_UNCHANGED)
        return floor_image, tile_image
    except requests.exceptions.RequestException as e:
        print(f"Load images Failed: {e}")
        return None
    
def create_tiled_texture(tile_img, target_size):
    try:
        small_tile = cv2.resize(tile_img, (300, 300), interpolation=cv2.INTER_AREA)
        texture = np.tile(small_tile, (target_size[1] // small_tile.shape[0] + 1, target_size[0] // small_tile.shape[1] + 1, 1))
        return texture[:target_size[1], :target_size[0]]
    except requests.exceptions.RequestException as e:
        print(f"Create tiled texture failed: {e}")
        return None
    

def apply_tiled_texture(original_image, mask, tile_image, floor_contour):
    try:
        tiled_texture = create_tiled_texture(tile_image, original_image.shape[:2])

        pts1 = np.float32([[0, 0], [tiled_texture.shape[1], 0], [tiled_texture.shape[1], tiled_texture.shape[0]], [0, tiled_texture.shape[0]]])
        pts2 = np.float32(cv2.boxPoints(cv2.minAreaRect(floor_contour)))
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        tiled_perspective = cv2.warpPerspective(tiled_texture, matrix, (original_image.shape[1], original_image.shape[0]))

        masked_floor = np.where(mask[:, :, None] == 255, tiled_perspective, original_image)

        alpha = 0.3
        final_floor = cv2.addWeighted(original_image, alpha, masked_floor, 1 - alpha, 0)

        return Image.fromarray(final_floor)
    except requests.exceptions.RequestException as e:
        print(f"Apply tiled texture failed: {e}")
        return None

def img_processing(floor_img_filename, tile_img_filename, save_filename='mask_floor.jpg'):
    try:
        api_response = query_api(floor_img_filename)

        if api_response is None:
            print("Hagging face API call failed Unable to proceed.")
            return
    except requests.exceptions.RequestException as e:
        print(f"Image processing failed: {e}")
        return None


    floor_image, tile_image = load_images(floor_img_filename, tile_img_filename)

    for item in api_response:
        label = item['label']
        if label == 'floor':
            mask_data = base64.b64decode(item['mask'])
            mask_image = Image.open(io.BytesIO(mask_data))
            mask_image = mask_image.resize(floor_image.shape[1::-1], Image.BILINEAR).convert('L')
            mask = np.array(mask_image)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            floor_contour = max(contours, key=cv2.contourArea)

            final_image = apply_tiled_texture(floor_image, mask, tile_image, floor_contour)

            final_image.save(save_filename)

def download_images(floorImage, tileImage):
    newpath = "Images/"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    try:
        # Downloading Image 1
        output1 = f"{newpath}floorImage.jpg"
        urllib.request.urlretrieve(floorImage, output1)
    except Exception as e:
        print("Error while downloading floorImage because ", e)

    try:
        # Downloading Image 2
        output2 = f"{newpath}tileImage.jpg"
        urllib.request.urlretrieve(tileImage, output2)
    except Exception as e:
        print("Error while downloading tileImage because ", e)

    return output1, output2

def upload_image(image_path):
    bucket = storage.bucket()
    now = datetime.datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    remote_path = f'mask_floor_{date_time}.jpg'

    try:
        # Upload the image
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(image_path)
        blob.make_public()
        img_url = blob.public_url
        print(f"Image '{image_path}' uploaded to '{img_url}' in default bucket.")
    except Exception as e:
        print(f"Error while uploading image: {e}")

    return img_url

def main_process(floorImage, tileImage):
    try:
        
        # Download images
        floor_img_path, tile_img_path = download_images(floorImage, tileImage)
        startTime = time.time()
        # Process images using the main function
        img_processing(floor_img_path, tile_img_path)
        endTime = time.time()
        totalTime = endTime - startTime
        print("Total time taken to preprocess the Image in Model:",totalTime)

        # Upload the processed image
        uploaded_image_url = upload_image('mask_floor.jpg')
        
        return uploaded_image_url
    
    except Exception as e:
        return {"Error in Main-Process Function:":e}
