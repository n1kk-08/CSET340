import tensorflow as tf
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Small
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np

# Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Load dataset using ImageDataGenerator (replace paths as needed)
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'dogs_dataset',  # path to dog breed dataset directory
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    'dogs_dataset',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes

def build_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Helper function to train and evaluate
def train_and_evaluate(model_name, base_model_class):
    print(f"\n🔧 Training {model_name}...")
    base_model = base_model_class(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet')
    model = build_model(base_model)

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        verbose=1
    )

    # Evaluation
    val_generator.reset()
    predictions = model.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    class_labels = list(val_generator.class_indices.keys())

    print(f"\n📊 Classification Report for {model_name}:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

# Train and evaluate each MobileNet version
train_and_evaluate("MobileNet V1", MobileNet)
train_and_evaluate("MobileNet V2", MobileNetV2)
train_and_evaluate("MobileNet V3 Small", MobileNetV3Small)
