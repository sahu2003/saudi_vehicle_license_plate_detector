from ultralytics import YOLO
import numpy as np
import cv2
import easyocr
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
import os

LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)
reader = easyocr.Reader(['ar', 'en'], gpu=False)

def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.readtext(license_plate_crop)

    width = img.shape[1]
    height = img.shape[0]

    if not detections:
        return None, None

    rectangle_size = license_plate_crop.shape[0] * license_plate_crop.shape[1]

    plate = []

    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > 0.17:
            bbox, text, score = result
            text = result[1]
            text = text.upper()
            scores += score
            plate.append(text)

    if len(plate) != 0:
        return " ".join(plate), scores / len(plate)
    else:
        return " ".join(plate), 0

# function to put Arabic text on an image
def put_arabic_text(image, text, position, font_path, font_size, color):
    # reshaped_text = arabic_reshaper.reshape(text)  # reshape the Arabic text (prevents words from being displayed as individual letters)
    bidi_text = get_display(text)  # reorder text for correct display

    #OpenCV image to a Pillow image 
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, font_size)  # Load the font

    # draw text on the Pillow image
    draw.text(position, bidi_text, font=font, fill=color)
    
    # convert back to OpenCV image
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return image

def detect_license(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            frame = cv2.imread(input_path)
            
            license_detections = license_plate_detector(frame)[0]
            print(license_detections)

            if len(license_detections.boxes.cls.tolist()) != 0:
                for license_plate in license_detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)

                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, frame)

                    if license_plate_text:
                        font_path = "arial.ttf"
                        font_size = 30
                        color = (0, 255, 0) #text colour

                        position = ((int(x1)), int(y1) - 30)
                        
                        frame = put_arabic_text(frame, license_plate_text, position, font_path, font_size, color)

            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, frame)
            print(f"Processed {filename} and saved to {output_path}")

input_directory = './images'
output_directory = './results'
detect_license(input_directory, output_directory)
