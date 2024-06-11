from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the ImageDataGenerator for training data
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Define the ImageDataGenerator for validation data (no augmentation)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Define the ImageDataGenerator for test data (no augmentation)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
