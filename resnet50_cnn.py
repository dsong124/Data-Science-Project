import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

image_size = (224, 224)
batch_size = 32
data_dir = 'Organized_Images'  # This folder contains Class_1, Class_2, ..., Class_8
'''
Class 1 - [1, 0, 0, 0, 0, 0, 0, 0]
Class 2 - [0, 1, 0, 0, 0, 0, 0, 0]
Class 3 - [0, 0, 1, 0, 0, 0, 0, 0]
Class 4 - [0, 0, 0, 1, 0, 0, 0, 0]
Class 5 - [0, 0, 0, 0, 1, 0, 0, 0]
Class 6 - [0, 0, 0, 0, 0, 1, 0, 0]
Class 7 - [0, 0, 0, 0, 0, 0, 1, 0]
Class 8 - [0, 0, 0, 0, 0, 0, 0, 1]

'''

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=42
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

#building the model 
input_shape = image_size + (3,)
num_classes = 8

base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("best_resnet50_fundus_model.h5", save_best_only=True)
]

#Training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks
)

#Evaluation
loss, accuracy = model.evaluate(val_ds)
print(f"\n Validation Accuracy: {accuracy * 100:.2f}%")
print(f" Validation Loss: {loss:.4f}")

model.save("resnet50_fundus_final_model.h5")

# accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# # loss
# plt.figure(figsize=(8, 6))
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()
