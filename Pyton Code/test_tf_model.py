import tensorflow as tf
import numpy as np
import os
from glob import glob
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image_path, target_size):
    image = Image.open(image_path)
    image = image.resize(target_size, Image.Resampling.NEAREST)
    image = image.convert('RGB')
    image = np.array(image).astype(np.uint8)
    return image

def load_test_data_with_labels(test_path, target_size, class_names):
    test_images = []
    test_labels = []
    for class_index, class_name in enumerate(class_names):
        class_folder_path = os.path.join(test_path, class_name)
        image_paths = glob(os.path.join(class_folder_path, '*.jpg'))
        print(f"Class '{class_name}': Found {len(image_paths)} images.")
        for image_path in image_paths:
            image = preprocess_image(image_path, target_size)
            test_images.append(image)
            test_labels.append(class_index)
    return np.array(test_images), np.array(test_labels)

def plot_confusion_matrix(cm, class_names, accuracy, precision, recall, f1):
    # Normalize the confusion matrix.
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percentages = cm_normalized * 100  # Convert to percentages

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentages, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    # Update the title to display all metrics on the same line
    title = f'Accuracy: {accuracy*100:.2f}% | Precision: {precision*100:.2f}% | Recall: {recall*100:.2f}% | F1 Score: {f1*100:.2f}%'
    
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def evaluate_model(interpreter, test_data, test_labels, class_names):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []
    for data in test_data:
        data = np.expand_dims(data, axis=0)
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data[0])
        predictions.append(prediction)

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = np.mean(predictions == test_labels)
    precision = precision_score(test_labels, predictions, average='macro', labels=range(len(class_names)))
    recall = recall_score(test_labels, predictions, average='macro', labels=range(len(class_names)))
    f1 = f1_score(test_labels, predictions, average='macro', labels=range(len(class_names)))
    
    # Calculate the confusion matrix
    cm = confusion_matrix(test_labels, predictions, labels=range(len(class_names)))
    
    # Plot the confusion matrix with additional metrics
    plot_confusion_matrix(cm, class_names, accuracy, precision, recall, f1)
    
    return accuracy, precision, recall, f1, cm

def main():
    model_path = r'D:\STUDY\PYTHON\TrainModelTrain\Experiment 1\SqueezeNet_V1.1\quantized_models\quantized_model.tflite'
    test_data_path = r'D:\STUDY\PYTHON\TrainModelTrain\Experiment 1\Model1_Test_Data'
    target_size = (128,128)
    class_names = ['Bacterial_Blight','Blast','Brownspot','Tungro']
    #[Bacterial_Blight, Blast, Brownspot, Tungro]
    #['Bacterial_Blight', 'Bacterial_Leaf_Streak', 'Bacterial_Panicle_Blight', 'Dead_Heart', 'Downy_Mildew', 'Hispa', 'normal', 'Blast', 'Brownspot', 'Tungro']

    print("Loading model...")
    interpreter = load_tflite_model(model_path)

    print("Loading test data...")
    test_data, test_labels = load_test_data_with_labels(test_data_path, target_size, class_names) 
    print(f"Loaded {len(test_data)} images for testing.") 
 
    print("Evaluating model...")
    accuracy, precision, recall, f1, cm = evaluate_model(interpreter, test_data, test_labels, class_names) 
    print(f"Model Accuracy: {accuracy * 100:.2f}%") 
    print(f"Model Precision: {precision * 100:.2f}%") 
    print(f"Model Recall: {recall * 100:.2f}%") 
    print(f"Model F1 Score: {f1 * 100:.2f}%")

if __name__ == "__main__":
    main()