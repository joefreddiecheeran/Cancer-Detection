import os
import numpy as np
import tensorflow as tf 
from models.model5 import create_model5
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import json 

IMG_HEIGHT, IMG_WIDTH = 150, 150  
BATCH_SIZE = 32

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'

train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False, 
)

model = create_model5()
model_name = 'model5'
print(f"Training {model_name}...")

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10, 
    steps_per_epoch=len(train_generator),
    validation_steps=len(test_generator)
)

model.save(f'models/{model_name}.h5')

history_dict = history.history
with open(f'reports/{model_name}_history.json', 'w') as f:
    json.dump(history_dict, f)

report = {
    'name': model_name,
    'train_accuracy': history.history['accuracy'][-1],
    'val_accuracy': history.history['val_accuracy'][-1],
    'train_loss': history.history['loss'][-1],
    'val_loss': history.history['val_loss'][-1],
}

test_steps = len(test_generator)
test_generator.reset()  
predictions = model.predict(test_generator, steps=test_steps)
predicted_classes = np.where(predictions > 0.5, 1, 0) 

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys()) 

report_path = f'reports/{model_name}_report.txt'
with open(report_path, 'w') as f:
    f.write(f"Train Accuracy: {report['train_accuracy']}\n")
    f.write(f"Validation Accuracy: {report['val_accuracy']}\n")
    f.write(f"Train Loss: {report['train_loss']}\n")
    f.write(f"Validation Loss: {report['val_loss']}\n")

    f.write("\nClassification Report:\n")
    f.write(classification_report(true_classes, predicted_classes, target_names=class_labels))

print(f"Report for {model_name} saved at {report_path}")
print(f"Model and history saved successfully!")
