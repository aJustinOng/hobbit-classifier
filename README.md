# Hobbit Face SVM Classifier ![](./assets/img/face-scan.svg) (README WIP)

**Skills:** `Python | NumPy | Pandas | Matplotlib | OpenCV | PyWavelets | scikit-learn | HTML | CSS | JavaScript`

**Tools:** `Jupyter Notebook | VS Code | PyCharm | Flask`

##### [See my other projects!](https://github.com/aJustinOng)

---

## Overview

This project is based on CodeBasic's [Sports Celebrity Image Classification](https://www.youtube.com/playlist?list=PLeo1K3hjS3uvaRHZLl-jLovIjBP14QTXc) project.

I was inspired to make this classification project when my friends started making plans to get together to watch the LOTR (Lord of the Rings) trilogy again. Hobbits are a race in the Tolkien franchise, and in the movies they are played by several well-known Hollywood actors such as Elijah Wood and Martin Freeman. I thought, since they are all male caucasian actors and played similar roles in the movies, can I build a model that can classify between them?

So I searched for and downloaded 50 images for each of the chosen five hobbit actors (Elijah Wood, Sean Astin, Billy Boyd, Dominic Monaghan, and Martin Freeman) on Google. For the preprocessing, I used OpenCV's Haar cascade classifiers to detect faces and eyes in those images, filtering out the unideal training images. I then stored the cropped facial regions into a separate folder before using PyWavelets to extract the facial regions from them. The combined images of both the original cropped image and Wavelet transformed image were split into train and test sets, which were finally used to train a SVM (support vector machine) model. I used GridSearchCV to determine the best model and parameters. After exporting the model as a Pickle file, I loaded it in a Flask server that was connected to a HTML/CSS/JavaScript webpage. The webpage allows the user to drop in an image to classify which of the five hobbits the image resembles. It also displays the confidence of the model and can detect multiple faces in a single image.

<img src="/assets/img/website-ui.gif" width="100%"/>

## Table of contents:
1. [Data Collection from Google](#1-data-collection-from-google)
2. [Importing Libraries and Data Loading](#2-importing-libraries-and-data-loading)
3. [Image Preprocessing: Data Cleaning](#3-image-preprocessing-data-cleaning)
4. [Image Preprocessing: Feature Engineering](#4-image-preprocessing-feature-engineering)
5. [Model Building Using SVM](#5-model-building-using-svm)
6. [Creating a Python Flask Server](#6-creating-a-python-flask-server)
7. [Creating a User-Friendly Webpage](#7-creating-a-user-friendly-webpage)
8. [Bonus: More Faces?](#8-bonus-more-faces)
9. [Summary](#summary)

## 1. Data Collection from Google

Since the theme of this project was hobbits, we will be classifying between five actors who have played the role in the Tolkien franchise: **Elijah Wood**, **Sean Astin**, **Billy Boyd**, **Dominic Monaghan**, and **Martin Freeman**. In the movies they played the hobbits Frodo Baggins, Samwise Gamgee, Peregrin (or Pippin) Took, Meriadoc (or Merry) Brandybuck, and Bilbo Baggins, respectively. Let us stick with the actors' names for the meantime. We will not worry too much about the hobbit names until we get to the webpage UI.

In an image classification project, a collection of good quality images of each classification object (in this case, faces of the actors) is required to train our model. I tested several image web scraping tools but they were not very effective being only able to scrap the image thumbnails on Google, resulting in poor quality images around 100px-200px in width/height. Web scraping has become more and more controlled in the recent years, so it may not be the way to go for this project. Since we are only classifying between five faces, I spent a couple hours manually clicking through and downloading 50 high quality images for each actor (250 images in total). Although tedious, this helps with the preprocessing as well since there will be few unsuitable training data (like faces of other people or obstructed faces). I placed these images in subfolders named after their respectively actors (e.g. path in repo: `model -> data -> dataset -> elijah_wood`)

> Note: If you have trouble in manually downloading images in a consistent .jpg or .png format, a image format converter browser extension is very helpful. They help minimize problems with difficult formats like WEBP files. Extensions also tend to come and go so any on the Chrome extension store should work well.

Some examples of the images we will be using:

`model/data/dataset/elijah_wood/19046_v9_bb.jpg:`

<img src="model/data/dataset/elijah_wood/19046_v9_bb.jpg" width="40%">

`model/data/dataset/elijah_wood/019c63_92e16961e1ca4d8589b03b451.jpg:`

<img src="model/data/dataset/elijah_wood/019c63_92e16961e1ca4d8589b03b451.jpg" width="60%">

## 2. Importing Libraries and Data Loading

### 2.1 Install Necessary Libraries

These are the required library installations and their specific versions I used for this project (written in `requirements.txt`:

```
PyWavelets==1.7.0
opencv-python==4.11.0
seaborn==0.8.1
```

We can quickly install these libraries in `requirements.txt` by using the following command in the command prompt:

```
pip install -r requirements.txt
```

### 2.2 Import Libraries

After successfully installing those libraries, we can create a new Jupyter Notebook and import the common libraries:

```
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
%matplotlib inline
```

### 2.3 Load Data

We can use `cv2` to read images and plot them. Let us load an image in the `test_images` folder:

```
img = cv2.imread('./data/test_images/image_1.jpg')
plt.imshow(img)
plt.show()
```

<img src="/assets/img/plt-elijah-wood-original.png" width="40%">

We will use this image to demonstrate our image preprocessing with data cleaning and feature engineering.

## 3. Image Preprocessing: Data Cleaning

### 3.1 Detect Facial Features with Haar Cascades

To ensure that we have good training data, we want images where the person's face can be seen clearly. A good indicator of a clearly-seen face is when both eyes can be detected by the model. We can use OpenCV Haar Cascades to do that.

Files can be downloaded [here](https://github.com/opencv/opencv/tree/master/data/haarcascades). We will keep these files under the path `/opencv/haarcascades`.

Here is a good [article](https://pyimagesearch.com/2021/04/12/opencv-haar-cascades) by Adrian Rosebrock about OpenCV Haar Cascades. Key takeaways are that OpenCV uses a sliding window method, and during that process, it computes different features. It functions on the fact that on an image of a face, it will have both lighter and darker areas (e.g. the eyes, the sides of the nose, mouth). It recognizes patterns such as:

- Eye regions tend to be darker than cheek regions.
- The nose region is brighter than the eye region.

Even though there are more modern methods to facial recognition, Haar Cascades are extremely fast and sufficient for a project like this. However, its main weakness is that it is prone to false positives, detecting faces when there is none. We can reduce the chances of false positives by picking images with both a face and two eyes.

Let us start by converting the image into grayscale:

```
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()
```

<img src="/assets/img/plt-elijah-wood-gray.png" width="40%">

We then import the pretrained Haar Cascade data under `face_cascade` and `eye_cascade`.

```
face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')
```

First, use `face_cascade` to attempt to detect a face in the image. We can draw a red box to visualize it.

```
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
(x, y, w, h) = faces[0]

cv2.destroyAllWindows()
face_img = cv2.rectangle(img,(x, y), (x + w, y + h),(255, 0, 0), 2)
plt.imshow(face_img)
plt.show()
```

<img src="/assets/img/plt-elijah-wood-face.png" width="40%">

It detects a face as expected. We can do the same for each eye.

```
cv2.destroyAllWindows()
for (x, y, w, h) in faces:
    face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = face_img[y: y + h, x: x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        
plt.figure()
plt.imshow(face_img, cmap='gray')
plt.show()
```

<img src="/assets/img/plt-elijah-wood-face-eyes.png" width="40%">

### 3.2 Crop the Facial Region of Image

The model is not interested in the rest of the image, so we can crop the image to just the area of the face.

```
%matplotlib inline
plt.imshow(roi_color, cmap='gray')
plt.show()
```

<img src="/assets/img/plt-elijah-wood-cropped-face-eyes.png" width="50%">

### 3.3 Crop Images if More Than Two Eyes

We can now write a function that returns a cropped image of the facial region, only if the it detects two or more eyes. We do two or more eyes is because it is possible for an image to contain multiple faces, and this model is also able to classify multiple faces.

```
def get_cropped_image_if_2_eyes (image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = img[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
```

```
original_image = cv2.imread('./data/test_images/image_1.jpg')
plt.imshow(original_image)
plt.show()
```

<img src="/assets/img/plt-elijah-wood-original.png" width="40%">

If we pass the previous image into the function, it returns the cropped image as expected.

```
cropped_img = get_cropped_image_if_2_eyes('./data/test_images/image_1.jpg')
plt.imshow(cropped_img)
plt.show()
```

<img src="/assets/img/plt-elijah-wood-cropped.png" width="50%">

What about a image where the face or eyes are obstructed?

```
orginal_image_obstructed = cv2.imread('./data/test_images/image_2.jpg')
plt.imshow(orginal_image_obstructed)
plt.show()
```

<img src="/assets/img/plt-elijah-wood-obstructed.png" width="60%">

The function does not return anything, which is what we want.

```
cropped_image_no_2_eyes = get_cropped_image_if_2_eyes('./data/test_images/image_2.jpg')
cropped_image_no_2_eyes
```

### 3.4 Save Cropped Images

Now we can perform this process on the actual dataset and save the cropped images to a separate path. Declare the paths and use `os` to find the directory for each person.

```
path_to_data = "./data/dataset/"
path_to_cropped_data = "./data/cropped_images/"
```

```
import os
img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)

img_dirs
```

<img src="/assets/img/jupyter-output-6.png" width="30%">

Now use `os` to create directories to store cropped images for each person. We can use `shutil` to remove existing directories to avoid complications. 

```
import shutil
if os.path.exists(path_to_cropped_data):
    shutil.rmtree(path_to_cropped_data)
os.mkdir(path_to_cropped_data)
```

Our directories and paths are ready so we can now run the `get_cropped_image_if_2_eyes()` function on every image in our dataset.

```
cropped_image_dirs = []
hobbit_file_names_dict = {}
for img_dir in img_dirs:
    count = 1
    hobbit_name = img_dir.split('/')[-1]
    hobbit_file_names_dict[hobbit_name] = []
    for entry in os.scandir(img_dir):
        roi_color = get_cropped_image_if_2_eyes(entry.path)
        if roi_color is not None:
            cropped_folder = path_to_cropped_data + hobbit_name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder)
                print("Generating cropped images in folder: ", cropped_folder)
            cropped_file_name = hobbit_name + str(count) + ".png"
            cropped_file_path = cropped_folder + "/" + cropped_file_name
            cv2.imwrite(cropped_file_path, roi_color)
            hobbit_file_names_dict[hobbit_name].append(cropped_file_path)
            count += 1
```
<img src="/assets/img/jupyter-output-7.png" width="70%">

## 4. Image Preprocessing: Feature Engineering

## 5. Model Building Using SVM

## 6. Creating a Python Flask Server

## 7. Creating a User-Friendly Webpage

## 8. Bonus: More Faces?

## Summary

