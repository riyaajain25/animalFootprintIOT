import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# =========================
# CONFIGURATION
# =========================
data_dir = r"C:\Users\HP\Desktop\animalFootprintIOT\train"
split_dir = r"C:\Users\HP\Desktop\animalFootPrintIOT\train_split"
img_size = (224, 224)
batch_size = 32
epochs_initial = 20
epochs_finetune = 50
num_unfreeze_layers = 50
USE_CLASS_WEIGHTS = True   # ✅ use class weights to handle imbalance

# =========================
# AUTO SPLIT (80% train, 20% val)
# =========================
print("📂 Preparing stratified data split...")

try:
    import splitfolders
except ImportError:
    os.system("pip install split-folders")
    import splitfolders

if not os.path.exists(os.path.join(split_dir, "train")):
    splitfolders.ratio(
        data_dir,
        output=split_dir,
        seed=42,
        ratio=(0.8, 0.2)
    )
    print("✅ Split created at:", split_dir)
else:
    print("ℹ️ Existing split found. Skipping re-split.")

# =========================
# LOAD DATASETS
# =========================
print("\n📦 Loading datasets...")

train_ds = image_dataset_from_directory(
    os.path.join(split_dir, "train"),
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)

val_ds_metrics = image_dataset_from_directory(
    os.path.join(split_dir, "val"),
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

class_names = train_ds.class_names
print("✅ Detected Classes:", class_names)

# Extract labels for class weighting
train_labels = np.concatenate([y.numpy() for x, y in train_ds])

# Prefetch for better performance
train_ds_fit = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds_fit = val_ds_metrics.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# =========================
# CLASS WEIGHTS (optional)
# =========================
class_weights = None
if USE_CLASS_WEIGHTS:
    print("\n⚖️ Calculating class weights...")
    classes = np.unique(train_labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_labels
    )
    class_weights = dict(zip(classes, weights))
    print("✅ Computed class weights:", class_weights)

# =========================
# MODEL ARCHITECTURE
# =========================
print("\n🔧 Building model...")

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.25),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(0.15, 0.15)
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet',
    alpha=1.0   # stronger backbone
)
base_model.trainable = False

inputs = layers.Input(shape=img_size + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = models.Model(inputs, outputs)

# =========================
# STAGE 1: INITIAL TRAINING
# =========================
print("\n🚀 Stage 1: Training top layers...")

initial_learning_rate = 1e-3
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=int(len(train_ds_fit) * 5),
    decay_rate=0.9,
    staircase=True
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

callback_initial = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=4,
    restore_best_weights=True
)

history_initial = model.fit(
    train_ds_fit,
    validation_data=val_ds_fit,
    epochs=epochs_initial,
    callbacks=[callback_initial],
    class_weight=class_weights
)

# =========================
# STAGE 2: FINE-TUNING
# =========================
print(f"\n🔁 Stage 2: Fine-tuning last {num_unfreeze_layers} layers...")

base_model.trainable = True
for layer in base_model.layers[:-num_unfreeze_layers]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

callback_finetune = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

start_epoch = history_initial.epoch[-1] + 1
history_finetune = model.fit(
    train_ds_fit,
    validation_data=val_ds_fit,
    epochs=epochs_finetune,
    callbacks=[callback_finetune],
    initial_epoch=start_epoch,
    class_weight=class_weights
)

# =========================
# SAVE CLEAN MODEL
# =========================
inference_model = tf.keras.Model(inputs=model.input, outputs=model.output)
inference_model.save("animal_footprint_best.keras")
print("✅ Model saved as 'animal_footprint_best.keras'")

# =========================
# EVALUATION & REPORTS
# =========================
print("\n📊 Evaluating model...")

val_metrics = model.evaluate(val_ds_metrics)
print(f"\n✅ Final Validation Loss: {val_metrics[0]:.4f}")
print(f"✅ Final Validation Accuracy: {val_metrics[1]:.4f}")

y_true = np.concatenate([y.numpy() for x, y in val_ds_metrics])
y_pred_probs = model.predict(val_ds_metrics)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

print("\n" + "="*60)
print("Classification Report (Precision, Recall, F1-Score)")
print("="*60)
print(classification_report(
    y_true,
    y_pred_labels,
    target_names=class_names,
    zero_division=0,
    digits=3
))

print("\n" + "="*60)
print("Confusion Matrix")
print("="*60)
cm = confusion_matrix(y_true, y_pred_labels)
print(cm)

per_class_acc = cm.diagonal() / cm.sum(axis=1)
for i, cls in enumerate(class_names):
    print(f"{cls}: {per_class_acc[i]*100:.2f}% accuracy")

# =========================
# PLOT TRAINING CURVES
# =========================
def combine_histories(h1, h2):
    combined = {}
    for k in h1.history.keys():
        combined[k] = h1.history.get(k, []) + h2.history.get(k, [])
    h = tf.keras.callbacks.History()
    h.history = combined
    return h

def plot_combined_history(h1, h2):
    hist = combine_histories(h1, h2)
    plt.figure(figsize=(10, 5))
    plt.plot(hist.history.get('accuracy', []), label='Train Accuracy')
    plt.plot(hist.history.get('val_accuracy', []), label='Validation Accuracy')
    plt.axvline(x=len(h1.history.get('accuracy', [])) - 1, color='r', linestyle='--', label='Fine-tuning Start')
    plt.title('Training + Fine-tuning Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_combined_history(history_initial, history_finetune)
print("\n🎉 Training complete!")
