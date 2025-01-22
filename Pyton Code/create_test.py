import os
import random
import shutil

def randomly_move_images(source_dir, destination_dir, num_images=200):
    # Make sure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # List all subdirectories in the source directory
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            # List all images in the folder
            all_images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            # Randomly select num_images images
            selected_images = random.sample(all_images, min(num_images, len(all_images)))

            # Destination folder for the selected images
            dest_folder = os.path.join(destination_dir, folder_name)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)

            # Move selected images to the destination directory
            for image in selected_images:
                src_path = os.path.join(folder_path, image)
                dest_path = os.path.join(dest_folder, image)
                shutil.move(src_path, dest_path)

# Example usage
source_directory = r'D:\STUDY\PYTHON\TrainModel\image_classification\scripts\training\datasets\data_model_1'  
destination_directory = r'D:\STUDY\PYTHON\TrainModel\image_classification\scripts\training\datasets\test'  
randomly_move_images(source_directory, destination_directory)
