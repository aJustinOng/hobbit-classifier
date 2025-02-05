import base64
import cv2
import pickle
import json
import numpy as np
from tensorflow.keras.models import load_model
from decimal import Decimal, ROUND_HALF_UP

__class_name_to_number = {}
__class_number_to_name = {}
__model = None

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/hobbit_model_v2.pkl', 'rb') as f:
            __model = pickle.load(f)

    print("loading saved artifacts...done")

def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []

    for img in imgs:
        img_resized = cv2.resize(img, (64, 64))
        img_normalized = img_resized / 255.0 
        img_reshaped = np.expand_dims(img_normalized, axis=0)

        predictions = __model.predict(img_reshaped)
        predicted_class = np.argmax(predictions[0])  # Get highest probability class
        class_probability = [
        float(Decimal(float(prob * 100)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        for prob in predictions[0]
        ]
        print(class_probability)

        result.append({
            'class': class_number_to_name(predicted_class),
            'class_probability': class_probability,
            'class_dictionary': __class_name_to_number
        })

    return result

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def get_b64_test_image_for_virat():
    with open("b64.txt") as f:
        return f.read()

def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = img[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces

if __name__ == "__main__":
    load_saved_artifacts()
    print(classify_image(None, "./test_images/elijah_wood.jpg"))
    print(classify_image(None, "./test_images/sean_astin.jpeg"))
    print(classify_image(None, "./test_images/billy_boyd.jpg"))
    print(classify_image(None, "./test_images/martin_freeman.jpg"))
