#densenet
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121

def create_model6():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False 

    model = tf.keras.models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')  
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
