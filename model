from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os



# Set your parameters
lr = 1e-4
epochs = 10
batch_size = 8
directory = "/content/drive/MyDrive/Face Mask Dataset/data"
categories = ["with_mask", "without_mask"]

data = []
labels = []

# Load and preprocess data
for category in categories:
    path = os.path.join(directory, category)
    for img in os.listdir(path)[:100]:
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(category)

# Convert to binary labels
labels = [1 if label == "with_mask" else 0 for label in labels]

data = np.array(data, dtype="float32")
labels = np.expand_dims(labels, axis=1)

# Train-test split
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=0)

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator()  # No augmentation for validation set

train_generator = train_datagen.flow(
    trainX,
    trainY,
    batch_size=batch_size,
    shuffle=True
)

test_generator = test_datagen.flow(
    testX,
    testY,
    batch_size=batch_size,
    shuffle=False
)

# Clear memory after loading images
del trainX, trainY

# Build the model
basemodel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

for layer in basemodel.layers:
    layer.trainable = False

headModel = basemodel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.3)(headModel)
headModel = Dense(1, activation="sigmoid")(headModel)

model = Model(inputs=basemodel.input, outputs=headModel)

# Compile the model
model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using the generator
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)


model.save("final_face_mask_detector.model",save_format = "h5")
