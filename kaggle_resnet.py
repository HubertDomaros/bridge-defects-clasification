import os

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np

import constants as c

# Data paths
data_path = "datasets/"
image_path = data_path + '/images'
train_imgs_dir = image_path + "/train"
test_imgs_dir = image_path + '/test'
val_imgs_dir = image_path + '/val'

# Load labels
label_dfs = {
    'train': '',
    'test': '',
    'val': ''
}

for split in label_dfs.keys():
    label_dfs[split] = pd.read_json(f'datasets/labels/{split}.json').drop(
        [c.BACKGROUND, c.CRACK, c.EFFLORESCENCE, 'Spallation_ExposedBars_CorrosionStain'], axis=1
    )

# Parameters
IMG_SIZE = 640
BATCH_SIZE = 8

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, labels_df, batch_size=32):
        self.image_dir = image_dir
        self.labels_df = labels_df
        self.batch_size = batch_size
        self.indices = range(len(self.labels_df))
        
    def __len__(self):
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        batch_labels = []
        
        for i in batch_indices:
            img_path = os.path.join(self.image_dir, self.labels_df.iloc[i]['img'])
            img = tf.keras.preprocessing.image.load_img(
                img_path, 
                target_size=(IMG_SIZE, IMG_SIZE)
            )
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = img / 255.0
            
            label = self.labels_df.iloc[i]['combination_id']
            batch_images.append(img)
            batch_labels.append(label)
            
        return np.array(batch_images), tf.keras.utils.to_categorical(batch_labels)

def create_data_generators():
    train_gen = CustomDataGenerator(train_imgs_dir, label_dfs['train'], BATCH_SIZE)
    val_gen = CustomDataGenerator(val_imgs_dir, label_dfs['val'], BATCH_SIZE)
    test_gen = CustomDataGenerator(test_imgs_dir, label_dfs['test'], BATCH_SIZE)
    return train_gen, val_gen, test_gen

# Rest of the code remains the same
def create_model():
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(np.unique(label_dfs['train']['combination_id'])), activation='softmax')
    ])

    return model

def train_model():
    train_generator, val_generator, test_generator = create_data_generators()
    
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2
            )
        ]
    )

    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    )

    return model, history, history_fine

if __name__ == "__main__":
    model, history, history_fine = train_model()
    model.save('pad_classification_model.h5')