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
8. [Summary](#summary)

## 1. Data Collection from Google

## 2. Importing Libraries and Data Loading

## 3. Image Preprocessing: Data Cleaning

## 4. Image Preprocessing: Feature Engineering

## 5. Model Building Using SVM

## 6. Creating a Python Flask Server

## 7. Creating a User-Friendly Webpage

## Summary

