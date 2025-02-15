import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns

# User should update this path to their own dataset location
DATA_DIR = 'path/to/your/HAM10000_metadata.csv'
data = pd.read_csv(DATA_DIR)

# Check the data
print(data.head())

# Mapping the diagnosis to numerical values
label_mapping = {
    'mel': 0,
    'nv': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6,
}
data['dx_num'] = data['dx'].map(label_mapping)

def load_images(img_ids, img_dir_base='path/to/your/HAM10000_images'):
    images = []
    for img_id in img_ids:
        img_path_part1 = os.path.join(img_dir_base, 'HAM10000_images_part_1', img_id + '.jpg')
        img_path_part2 = os.path.join(img_dir_base, 'HAM10000_images_part_2', img_id + '.jpg')

        # Check if the image exists in either part
        if os.path.exists(img_path_part1):
            img_path = img_path_part1
        elif os.path.exists(img_path_part2):
            img_path = img_path_part2
        else:
            # Print a warning instead of raising an error and skip the image
            print(f"Warning: Image not found: {img_id}.jpg. Skipping this image.")
            continue  # Skip to the next image

        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(72, 72))
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        images.append(img)

    return np.array(images)

# Create image dataset
X = load_images(data['image_id'].values)
y = data['dx_num'].values

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Configure hyperparameters for EfficientNetV2
num_classes = len(label_mapping)
image_size = 72
batch_size = 32
num_epochs = 12

# Build the EfficientNetV2 model
def create_efficientnet_model():
    base_model = keras.applications.EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    base_model.trainable = True  # Set to True to fine-tune the model

    inputs = layers.Input(shape=(image_size, image_size, 3))
    features = base_model(inputs, training=False)
    features = layers.GlobalAveragePooling2D()(features)
    outputs = layers.Dense(num_classes, activation='softmax')(features)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Compile, train, and evaluate the EfficientNetV2 model
def run_experiment(model):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=batch_size,
                        epochs=num_epochs,
                        callbacks=[early_stopping])

    return history

# Create and train the model
efficientnet_model = create_efficientnet_model()
history = run_experiment(efficientnet_model)

# Plot training history
def plot_history(history):
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

plot_history(history)

# Visualize Confusion Matrix
y_pred = np.argmax(efficientnet_model.predict(X_val), axis=1)
conf_matrix = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()