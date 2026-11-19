# 1. IMPORT LIBRARIES
import os, zipfile, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *
from tensorflow.keras import layers, models, callbacks

# 2. SETTINGS
SEED = 42
IMG_SIZE = 128
EPOCHS = 60
BATCH_SIZE = 16

tf.random.set_seed(SEED)
np.random.seed(SEED)

# 3. LOAD DATASET
with zipfile.ZipFile("/content/liver_disease_samples_100.zip") as z:
    names = z.namelist()
    file = next(n for n in names if n.endswith("labels.csv"))
    prefix = file[:-10]
    df = pd.read_csv(io.BytesIO(z.read(file)))

    X = np.array([
        np.array(Image.open(z.open(prefix + f))
        .convert("L").resize((IMG_SIZE, IMG_SIZE))) / 255
        for f in df["filename"]
    ]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y = LabelEncoder().fit_transform(df["label"])

# 4. SPLIT DATASET
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5,
    stratify=y_temp, random_state=SEED
)

# 5. BUILD CNN MODEL
model = models.Sequential([
    layers.Input(shape=(128, 128, 1)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),

    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.4),
    layers.Dense(64, activation="relu"),
    layers.Dense(4, activation="softmax")
])

# 6. COMPILE MODEL
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 7. CALLBACKS
cbs = [
    callbacks.EarlyStopping(
        monitor="val_loss", patience=12,
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5
    )
]

# 8. TRAIN MODEL
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cbs
)

# 9. EVALUATE MODEL
pred = np.argmax(model.predict(X_test), axis=1)

print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))

# 10. PLOT GRAPHS
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.legend()
plt.show()

plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.legend()
plt.show()

# 11. SAVE MODEL
model.save("liver_cnn.keras")
print("Model Saved Successfully")