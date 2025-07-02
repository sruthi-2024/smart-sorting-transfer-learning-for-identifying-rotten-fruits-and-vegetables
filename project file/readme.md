project.executable file
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
import os

# Set paths
dataset_path = "/path/to/dataset"  # Replace with your dataset path
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')

# Image generators
datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2,
                             horizontal_flip=True, shear_range=0.2)
train_gen = datagen.flow_from_directory(train_path, target_size=(224, 224),
                                        batch_size=32, class_mode='categorical')
val_gen = datagen.flow_from_directory(val_path, target_size=(224, 224),
                                      batch_size=32, class_mode='categorical')

# Load MobileNetV2 base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add custom head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
out = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=out)

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save model
model.save("fruit_veg_classifier.h5")
