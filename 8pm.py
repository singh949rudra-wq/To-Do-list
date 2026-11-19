#exp 8



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image

# 1. Settings
DATASET_PATH = "/content/Brain_Tumor_Dataset/brain_dataset_200_synthetic"
PREDICT_FOLDER = "/content/predict_images"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30

# 2. Load dataset
train = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="training",
    seed=42, image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, label_mode="binary")

val = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="validation",
    seed=42, image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, label_mode="binary")

print("Classes:", train.class_names)

# 3. Speed up data
AUTOTUNE = tf.data.AUTOTUNE
train = train.prefetch(AUTOTUNE)
val = val.prefetch(AUTOTUNE)

# 4. Data augmentation
augment = Sequential([
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.08),
    tf.keras.layers.RandomTranslation(0.05, 0.05)
])

# 5. Build CNN model
model = Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Rescaling(1./255),
    augment,

    Conv2D(32, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

# 6. Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(),
             tf.keras.metrics.Recall(),
             tf.keras.metrics.AUC()]
)

model.summary()

# 7. Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=7,
                  restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=3, min_lr=1e-7)
]

# 8. Train model
history = model.fit(
    train, validation_data=val,
    epochs=EPOCHS, callbacks=callbacks
)

# 9. Evaluate model
results = model.evaluate(val)
print("Evaluation:", results)

# 10. Plot accuracy
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 11. Plot loss
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 12. Save model
model.save("/content/brain_tumor_cnn.keras")
print("Model saved!")

# 13. Predict one MRI image
def predict_image(path):
    img = image.load_img(path, target_size=(224, 224))
    arr = np.expand_dims(image.img_to_array(img), axis=0)
    p = model.predict(arr, verbose=0)[0][0]

    if p >= 0.5:
        return "Tumor", p * 100, p
    return "Normal", (1-p) * 100, p

# 14. Predict all images in folder
results = []
extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

for name in sorted(os.listdir(PREDICT_FOLDER)):
    if name.lower().endswith(extensions):
        path = os.path.join(PREDICT_FOLDER, name)
        label, confidence, probability = predict_image(path)

        results.append({
            "Filename": name,
            "Prediction": label,
            "Confidence (%)": round(confidence, 2),
            "Tumor Probability": round(float(probability), 4)
        })

# 15. Display and save results
df = pd.DataFrame(results)
print("\nMRI PREDICTION RESULTS")
print(df.to_string(index=False))

df.to_csv("/content/brain_mri_predictions.csv", index=False)
print("Predictions saved!")

# 16. Prediction summary
print("\nTotal Images:", len(df))
print("Normal:", (df["Prediction"] == "Normal").sum())
print("Tumor:", (df["Prediction"] == "Tumor").sum())