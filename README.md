# Generative Deep Learning Projects

This repository contains a collection of projects based on the book "Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play, 1st Edition." The projects cover various aspects of generative deep learning, including image generation, text generation, and music composition.

Table of Contents
Introduction
Projects
[1. Project Title 1](https://github.com/sik247/Generative-Deep-Learning/tree/main/Chapter_2_Deeplearning)


# Introduction
Generative Deep Learning has emerged as a fascinating field, enabling machines to create new content in various domains. This repository is a collection of projects inspired by the concepts and techniques presented in the book "Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play, 1st Edition."

# Project 1 - CIFAR 

# CIFAR-10 Image Classification
This project focuses on implementing a neural network for image classification using the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, making it a suitable benchmark for testing image classification models.

Project Highlights:
Data Preparation:

The CIFAR-10 dataset is loaded and preprocessed.
Images are normalized to the range [0, 1], and labels are one-hot encoded.
Model Architecture:

A simple neural network is defined using the Keras API.
The model includes flatten and dense layers with ReLU activation functions.
The output layer uses softmax activation for multiclass classification.
Training the Model:

The model is compiled using the Adam optimizer and categorical cross-entropy loss.
Training is performed on the training set with a batch size of 32 for 10 epochs.
Model Evaluation:

The trained model is evaluated on the test set to assess its performance.
Prediction and Visualization:

Random test images are selected to make predictions.
The actual and predicted classes are displayed along with the corresponding images.

Usage instruction provided in readme of code*
