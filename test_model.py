from tensorflow.keras.models import load_model

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import argparse
import pickle
import cv2

import datasets

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 50
INIT_LR = 1e-1
BS = 128

# load the A-Z and MNIST datasets, respectively
print("[INFO] loading datasets...")
(azData, azLabels) = datasets.load_az_dataset("KaggleAZ/Kaggle_AZ_Data.csv")
(digitsData, digitsLabels) = datasets.load_mnist_dataset()

azLabels += 10

data = np.vstack([azData, digitsData])
labels = np.hstack([azLabels, digitsLabels])

data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")

data = np.expand_dims(data, axis=-1)
data /= 255.0

le = LabelBinarizer()
labels = le.fit_transform(labels)
counts = labels.sum(axis=0)

classTotals = labels.sum(axis=0)
classWeight = {}

for i in range(0, len(classTotals)):
	classWeight[i] = classTotals.max() / classTotals[i]

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# define the list of label names
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

# save the model to disk
print("[INFO] loading model...")
model = load_model("NMIST_Kaggle.model")

# demo the model design
# print("[INFO] model structure...")
# model.summary()

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))



# Make some charts of character evaluations
for itr in range(0, 10):
  images = []
  # randomly select a few testing characters
  for i in np.random.choice(np.arange(0, len(testY)), size=(49,)):
    # classify the character
    probs = model.predict(testX[np.newaxis, i])
    prediction = probs.argmax(axis=1)
    label = labelNames[prediction[0]]

    # extract the image from the test data and initialize the text
    # label color as green (correct)
    image = (testX[i] * 255).astype("uint8")
    color = (0, 255, 0)

    # otherwise, the class label prediction is incorrect
    if prediction[0] != np.argmax(testY[i]):
      color = (0, 0, 255)

    # merge the channels into one image, resize the image from 32x32
    # to 96x96 so we can better see it and then draw the predicted
    # label on the image
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
      color, 2)

    # add the image to our list of output images
    images.append(image)

  # construct the montage for the images
  montage = build_montages(images, (96, 96), (7, 7))[0]

  # show the output montage
  cv2.imshow("OCR Results", montage)
  cv2.waitKey(2000)
  # cv2.waitKey(0)

cv2.destroyAllWindows()
