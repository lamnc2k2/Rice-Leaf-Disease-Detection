import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set a seed for reproducibility
SEED = 1234
np.random.seed(SEED)

# Directory paths
original_data_dir = r'D:\STUDY\PYTHON\TrainModelTrain\ImageClassification\Deployment3\Model1\image_classification\scripts\training\datasets\SERIOUS\Experiment 5\Experiment5_Data'
augmented_data_dir = r'D:\STUDY\PYTHON\TrainModelTrain\ImageClassification\Deployment3\Model1\image_classification\scripts\training\datasets\SERIOUS\Experiment 5\Experiment5_Data'

# Paramater
target_size = (128, 128)  # Adjust based on your image dimensions
default_num_images = 2000  # Default number of images to generate per class if not specified in num_images_dict

# Image Data Generator with Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

def augment_images(class_name, target_num_images):
    class_dir = os.path.join(original_data_dir, class_name)
    save_dir = os.path.join(augmented_data_dir, class_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Count existing images
    existing_images_count = len([name for name in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, name))])
    images_to_generate = target_num_images - existing_images_count  # Adjust the number of images to generate
    
    if images_to_generate <= 0:
        print(f"Class {class_name} already has {existing_images_count} images, which meets or exceeds the target of {target_num_images}. No augmentation needed.")
        return
    
    generator = datagen.flow_from_directory(
        directory=original_data_dir,
        classes=[class_name],
        target_size=target_size,
        batch_size=1,
        save_to_dir=save_dir,
        save_prefix=f"{class_name}_",
        save_format='jpg',
        seed=SEED
    )

    for _ in range(images_to_generate):
        generator.next()
        # No need for manual file renaming, as the save_prefix and timestamp ensure uniqueness

# Dictionary for number of images per class (example format)
num_images_dict = {}

for class_name in os.listdir(original_data_dir):
    if os.path.isdir(os.path.join(original_data_dir, class_name)):
        target_num_images = num_images_dict.get(class_name, default_num_images)
        print(f"Processing class: {class_name} to reach a total of {target_num_images} images.")
        try:
            augment_images(class_name, target_num_images)
        except Exception as e:
            print(f"Error processing class {class_name}: {e}")

print("Data augmentation completed.")
