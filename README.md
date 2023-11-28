CIFAR-10 Image Classification with Neural Network
This repository contains a Python implementation of a neural network for image classification using the CIFAR-10 dataset. The CIFAR-10 dataset is a collection of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.

Overview
The project is structured into sections covering data preparation, model building, training, evaluation, and visualization of predictions on random test images.

Requirements
Make sure you have the following installed:

Python 3.x
TensorFlow 2.x
NumPy
Matplotlib
You can install the required dependencies using the following command:

bash
Copy code
pip install numpy matplotlib tensorflow
Setup
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/your-repo.git
cd your-repo
Open the project in your preferred Python environment.

Usage
1. Prepare Data
In this section, we load and preprocess the CIFAR-10 dataset.

python
Copy code
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, utils, datasets

# Number of classes in CIFAR-10
NUM_CLASSES = 10

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode the labels
y_train = utils.to_categorical(y_train, NUM_CLASSES)
y_test = utils.to_categorical(y_test, NUM_CLASSES)
2. Build Model
Define and build the neural network model.

python
Copy code
input_layer = layers.Input((32, 32, 3))

x = layers.Flatten()(input_layer)
x = layers.Dense(200, activation="relu")(x)
x = layers.Dense(150, activation="relu")(x)

output_layer = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(input_layer, output_layer)

model.summary()
3. Train Model
Compile and train the model on the training data.

python
Copy code
opt = optimizers.Adam(learning_rate=0.0005)
model.compile(
    loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True)
4. Evaluate Model
Evaluate the model on the test data.

python
Copy code
model.evaluate(x_test, y_test)
5. Prediction and Visualization
Make predictions on random test images and visualize the results.

python
Copy code
# Define class labels
CLASSES = np.array([
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
])

# Make predictions on test data
preds = model.predict(x_test)
preds_single = CLASSES[np.argmax(preds, axis=-1)]
actual_single = CLASSES[np.argmax(y_test, axis=-1)]

# Visualize random test images and predictions
n_to_show = 10
indices = np.random.choice(range(len(x_test)), n_to_show)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i + 1)
    ax.axis("off")
    ax.text(
        0.5,
        -0.35,
        "pred = " + str(preds_single[idx]),
        fontsize=10,
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        -0.7,
        "act = " + str(actual_single[idx]),
        fontsize=10,
        ha="center",
        transform=ax.transAxes,
    )
    ax.imshow(img)

plt.show()
