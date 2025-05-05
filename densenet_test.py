import tensorflow as tf
import os
import numpy as np
import cv2

best_model_file = "/Users/danielsong/Desktop/DS Project/densenet201_fundus_final_model.h5"
model = tf.keras.models.load_model(best_model_file)
print(model.summary())

input_shape = (224, 224)
batch_size = 16
path = "/Users/danielsong/Desktop/DS Project/Organized_Images"


categories = os.listdir(path)
categories.sort()

print(categories)
print(len(categories))

def prepareImage(img):
    resized = cv2.resize(img, input_shape, interpolation=cv2.INTER_AREA)
    imgResult = np.expand_dims(resized, axis=0)
    imgResult = imgResult / 255.
    return imgResult

testImagePath = "/Users/danielsong/Desktop/DS Project/Organized_Images/Class_3/95_left.jpg"
img = cv2.imread(testImagePath)

ImageForModel = prepareImage(img)

# run prediction
result = model.predict(ImageForModel, verbose=1)
print(result)

answers = np.argmax(result, axis=1)
print(answers)

text = categories[answers[0]]
print("The predicted class is : " + text)

font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text, (20, 20), font, 1, (0, 255, 255), 2)  # Yellow color
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
