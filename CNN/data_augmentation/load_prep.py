# Set up directories
train_dir = 'dataset/train'
val_dir = 'dataset/validation'
test_dir = 'dataset/test'

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),  # Image size
    batch_size=32,
    class_mode='binary'  # use 'categorical' for multi-class classification
)

# Load validation data
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(64, 64),  # Image size
    batch_size=32,
    class_mode='binary'  # use 'categorical' for multi-class classification
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),  # Image size
    batch_size=32,
    class_mode='binary'  # use 'categorical' for multi-class classification
)
