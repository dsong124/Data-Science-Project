import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

DATA_DIR    = 'Organized_Images'     
IMG_HEIGHT  = 224
IMG_WIDTH   = 224
BATCH_SIZE  = 16
EPOCHS      = 30
LR          = 1e-4
VALID_SPLIT = 0.2
SEED        = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    validation_split=VALID_SPLIT,
    subset='training',
    seed=SEED
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    validation_split=VALID_SPLIT,
    subset='validation',
    seed=SEED
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes (in order):", class_names)

# prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# model
base_model = DenseNet201(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print(model.summary())

best_model_file = "/Users/danielsong/Desktop/DS Project/e:/temp/densenet201_fundus_final_model.h5"
callbacks = [
    ModelCheckpoint(best_model_file,
                    monitor="val_accuracy",
                    verbose=1,
                    save_best_only=True),
    ReduceLROnPlateau(monitor="val_accuracy",
                      patience=5,
                      factor=0.1,
                      min_lr=1e-6,
                      verbose=1),
    EarlyStopping(monitor="val_accuracy",
                  patience=20,
                  verbose=1)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

best_val_acc = max(history.history['val_accuracy'])
print(f"Highest Validation Accuracy: {best_val_acc:.4f}")
