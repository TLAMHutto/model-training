# Load and preprocess a single image
from tensorflow.keras.preprocessing import image

img_path = 'path/to/single_image.jpg'
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make prediction
prediction = cnn_model.predict(img_array)
print('Prediction:', prediction)
