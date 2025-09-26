# Import standard numerical libraries
import numpy as np # array operations and numerics
import matplotlib.pyplot as plt # plotting library
from IPython.display import display, Markdown # helps print Markdown-style headings in Colab


# Import tensorflow and keras modules
import tensorflow as tf # TensorFlow main package
from tensorflow import keras # Keras high-level API (bundled with TF)
from tensorflow.keras import layers, models # common submodules for building models

# Function to show markdown headings
def md(text):
  """Display markdown text in notebook outputs for section titles."""
  display(Markdown(text))


# Function to plot a grid of images with labels
def plot_image_grid(images, labels, preds=None, class_names=None, num=9, title=None):
  """Plot `num` images in a square grid with actual and optionally predicted labels.


  images: array-like of images (H,W) or (H,W,1)
  labels: ground-truth integer labels
  preds: optional predicted integer labels (same shape as labels)
  class_names: list of label names indexed by integer label
  num: number of images to show (square)
  title: optional grid title
  """
  # Ensure num is a perfect square for grid layout
  sq = int(np.ceil(np.sqrt(num)))
  plt.figure(figsize=(sq * 2.2, sq * 2.2))
  if title:
    plt.suptitle(title)
  for i in range(num):
    plt.subplot(sq, sq, i + 1)
    img = images[i]
    # If image has channel dim, squeeze it
    if img.ndim == 3 and img.shape[2] == 1:
      img = img.squeeze(-1)
    plt.imshow(img, cmap='gray')
    true_label = class_names[labels[i]] if class_names is not None else str(labels[i])
    if preds is None:
      plt.title(f"GT: {true_label}")
    else:
      pred_label = class_names[preds[i]] if class_names is not None else str(preds[i])
      plt.title(f"GT: {true_label}\nPred: {pred_label}")
    plt.axis('off')
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.show()

  md('## Section 2 — Load Fashion MNIST')
# Load dataset directly from Keras datasets (downloads automatically in Colab if missing)
# This returns tuples: (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()


# Print dataset shapes and a quick sanity check
print('# Training images:', x_train.shape) # expected (60000, 28, 28)
print('# Training labels:', y_train.shape) # expected (60000,)
print('# Test images:', x_test.shape) # expected (10000, 28, 28)
print('# Test labels:', y_test.shape) # expected (10000,)

# ---------------------------------------------------------
# Show 10 images (1 per class, labels 0–9) BEFORE mapping
# ---------------------------------------------------------

# Find the first occurrence of each label (0–9) in training set
unique_labels = np.arange(10)
indices = [np.where(y_train == label)[0][0] for label in unique_labels]

# Plot in 2 rows × 5 columns
plt.figure(figsize=(12, 5))
for i, idx in enumerate(indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[idx], cmap='gray')
    plt.title(f"Label: {y_train[idx]}")  # numeric label only
    plt.axis('off')

plt.suptitle("One example from each label (0–9) BEFORE defining class names", fontsize=14)
plt.show()

# ---------------------------------------------------------
# Fashion MNIST Class Labels (Mapping from integers to names)
# ---------------------------------------------------------
# The Fashion MNIST dataset provides labels as integers (0–9).
# Each integer corresponds to a specific category of clothing item.
# Using human-readable names makes evaluation and visualization
# much easier (e.g., in predictions, confusion matrix, and plots).
#
# Mapping:
#  0 -> T-shirt/top
#  1 -> Trouser
#  2 -> Pullover
#  3 -> Dress
#  4 -> Coat
#  5 -> Sandal
#  6 -> Shirt
#  7 -> Sneaker
#  8 -> Bag
#  9 -> Ankle boot
#
# Note: This mapping is fixed and defined by the creators of Fashion MNIST.
#       It is the same every time you load the dataset.
# ---------------------------------------------------------

class_names = [
    'T-shirt/top',  # class 0
    'Trouser',      # class 1
    'Pullover',     # class 2
    'Dress',        # class 3
    'Coat',         # class 4
    'Sandal',       # class 5
    'Shirt',        # class 6
    'Sneaker',      # class 7
    'Bag',          # class 8
    'Ankle boot'    # class 9
]

print("### Fashion MNIST Label Mapping (0–9):")
for i, name in enumerate(class_names):
    print(f"Label {i} → {name}")

md('## Section 3 — Exploratory Data Analysis (EDA)')


# Show first 9 images with their labels
print('\n### Sample images from training set (first 9):')
plot_image_grid(x_train, y_train, class_names=class_names, num=9, title='First 9 training images')


# Show class distribution on the training set
import collections
train_counts = collections.Counter(y_train)
classes = list(range(10))
counts = [train_counts[c] for c in classes]

# ---------------------------------------------------------
# Remark if dataset is balanced
# ---------------------------------------------------------
print("\n### Training set class counts:")
for label, count in zip(classes, counts):
    print(f"Label {label} ({class_names[label]}): {count}")


plt.figure(figsize=(8,4))
plt.bar(classes, counts)
plt.xticks(classes, class_names, rotation=45, ha='right')
plt.ylabel('Number of images')
plt.title('Class distribution in training set')
plt.tight_layout()
plt.show()

# Simple balance check
min_count, max_count = min(counts), max(counts)
if max_count - min_count <= 500:
    print("\nRemark: The dataset is fairly balanced — each class has ~6,000 samples.")
else:
    print("\nRemark: The dataset shows imbalance between classes.")


# Show pixel value statistics to inform normalization
print('\n### Pixel statistics before normalization:')
print('Min pixel value:', x_train.min())
print('Max pixel value:', x_train.max())
print('Mean pixel value:', x_train.mean().round(3))
print('Std pixel value:', x_train.std().round(3))

md('## Section 4 — Preprocessing')


# Normalize images to [0,1] range by dividing by 255.0 (float32 conversion)
# Keep copies for NN (flattened) and CNN (with channel dim)
x_train_norm = x_train.astype('float32') / 255.0 # normalized training images
x_test_norm = x_test.astype('float32') / 255.0 # normalized test images


# Prepare data for simple Neural Network (NN): flatten 28x28 -> 784 vector
# We will keep a separate variable to avoid accidental reuse
x_train_nn = x_train_norm.reshape((-1, 28*28)) # shape (60000, 784)
x_test_nn = x_test_norm.reshape((-1, 28*28)) # shape (10000, 784)


# Prepare data for Convolutional Neural Network (CNN): add explicit channel dim -> (28,28,1)
x_train_cnn = x_train_norm.reshape((-1, 28, 28, 1)) # shape (60000, 28, 28, 1)
x_test_cnn = x_test_norm.reshape((-1, 28, 28, 1)) # shape (10000, 28, 28, 1)


# Print shapes to confirm
print('\nShapes after preprocessing:')
print('x_train_nn:', x_train_nn.shape)
print('x_test_nn :', x_test_nn.shape)
print('x_train_cnn:', x_train_cnn.shape)
print('x_test_cnn :', x_test_cnn.shape)

md('## Section 5 — Create validation split (from training data)')


# Using the `validation_split` argument during .fit() instead of manual split.
# This keeps code concise and consistent between models. We'll use 10% as validation.
val_split = 0.1
print('Using validation split:', val_split)

md('## Section 6 — Neural Network (Dense-only)')
print('\n### 6.1 — Build model architecture (NN)')


# Building a simple fully connected neural network.
# Architecture notes (explanatory):
# Input layer: 784-dim flattened vector
# Dense layer (ReLU): 128 neurons, activation = ReLU (non-linear)
# Dense layer (ReLU): 64 neurons
# Output layer: 10 neurons with softmax activation for multiclass probability output


nn_model = models.Sequential(name='simple_NN')
# Input is flattened 28*28 vector; include input_shape to allow model.summary()
nn_model.add(layers.Input(shape=(28*28,), name='input_flat')) # input layer
nn_model.add(layers.Dense(128, activation='relu', name='dense_128')) # first hidden dense layer
nn_model.add(layers.Dense(64, activation='relu', name='dense_64')) # second hidden dense layer
nn_model.add(layers.Dense(10, activation='softmax', name='output')) # softmax output for 10 classes


# Print architecture summary (model layers and parameters)
nn_model.summary()


print('\n### 6.2 — Compile model (loss, optimizer, metrics)')
# Use Adam optimizer (adaptive), sparse_categorical_crossentropy because labels are integer encoded
nn_model.compile(
  optimizer='adam', # Adam optimizer with default learning rate
  loss='sparse_categorical_crossentropy', # appropriate for integer labels (not one-hot)
  metrics=['accuracy'] # track accuracy during training and evaluation
)


print('\n### 6.3 — Train NN model')
# Train for a modest number of epochs; you can increase epochs in Colab for better performance
nn_epochs = 12 # number of epochs for NN training
nn_batch_size = 128 # batch size for gradient updates


# Fit the model. validation_split uses a portion of the training set for validation each epoch.
history_nn = nn_model.fit(
  x_train_nn, y_train, # training inputs and labels for NN
  epochs=nn_epochs,
  batch_size=nn_batch_size,
  validation_split=val_split,
  verbose=2
)

md('## Section 7 — Evaluate Neural Network on Test Set')


nn_test_loss, nn_test_acc = nn_model.evaluate(x_test_nn, y_test, verbose=2)
print(f"NN Test loss: {nn_test_loss:.4f}")
print(f"NN Test accuracy: {nn_test_acc:.4f}")

md('## Section 8 — NN: Sample predictions (5 images)')
# Choose 5 sample images from test set to show predictions
num_samples = 5
sample_idx = np.random.choice(len(x_test_nn), size=num_samples, replace=False)
# Get images for plotting (original grayscale images) and predictions
sample_images = x_test[sample_idx] # original 28x28 images (un-normalized but fine for display)
# Predict probabilities and then convert to predicted class indices
pred_probs_nn = nn_model.predict(x_test_nn[sample_idx]) # shape (num_samples, 10)
pred_labels_nn = np.argmax(pred_probs_nn, axis=1)


# Plot the selected sample images with predicted and actual labels
plot_image_grid(sample_images, y_test[sample_idx], preds=pred_labels_nn, class_names=class_names, num=num_samples, title='NN sample predictions')

md('## Section 9 — Convolutional Neural Network (CNN)')
print('\n### 9.1 — Build model architecture (CNN)')


# CNN architecture notes (explanatory):
# - Conv2D + ReLU layers extract local spatial features using learnable filters
# - MaxPooling reduces spatial resolution (and parameters), adds translation invariance
# - Flatten and Dense layers convert features to class predictions
# A reasonably small CNN that works well on Fashion MNIST:


cnn_model = models.Sequential(name='simple_CNN')
cnn_model.add(layers.Input(shape=(28,28,1), name='input_image')) # input for CNN: height,width,channels
cnn_model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', name='conv2d_1')) # conv layer 32 filters
cnn_model.add(layers.MaxPooling2D((2,2), name='maxpool_1')) # reduce spatial dimensions by factor of 2
cnn_model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv2d_2')) # conv layer 64 filters
cnn_model.add(layers.MaxPooling2D((2,2), name='maxpool_2'))
cnn_model.add(layers.Flatten(name='flatten')) # flatten feature maps to vector
cnn_model.add(layers.Dense(128, activation='relu', name='dense_128')) # fully connected layer
cnn_model.add(layers.Dropout(0.4, name='dropout')) # dropout for regularization
cnn_model.add(layers.Dense(10, activation='softmax', name='output')) # final classification layer


# Print model summary
cnn_model.summary()


print('\n### 9.2 — Compile CNN (loss, optimizer, metrics)')
cnn_model.compile(
  optimizer='adam', # Adam optimizer
  loss='sparse_categorical_crossentropy', # integer labels
  metrics=['accuracy']
)


print('\n### 9.3 — Train CNN model')
cnn_epochs = 12 # number of epochs for CNN (same as NN for fair comparison)
cnn_batch_size = 128


history_cnn = cnn_model.fit(
  x_train_cnn, y_train,
  epochs=cnn_epochs,
  batch_size=cnn_batch_size,
  validation_split=val_split,
  verbose=2
)

md('## Section 10 — Evaluate CNN on Test Set')
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(x_test_cnn, y_test, verbose=2)
print(f"CNN Test loss: {cnn_test_loss:.4f}")
print(f"CNN Test accuracy: {cnn_test_acc:.4f}")

md('## Section 11 — CNN: Sample predictions (5 images)')
# Use the same sample indices as NN to compare predictions on identical images
pred_probs_cnn = cnn_model.predict(x_test_cnn[sample_idx])
pred_labels_cnn = np.argmax(pred_probs_cnn, axis=1)


plot_image_grid(sample_images, y_test[sample_idx], preds=pred_labels_cnn, class_names=class_names, num=num_samples, title='CNN sample predictions')

md('## Section 12 — Comparison and Training Curves')
print(f"NN test accuracy: {nn_test_acc:.4f}")
print(f"CNN test accuracy: {cnn_test_acc:.4f}")


# Calculate percentage improvement
improvement = (cnn_test_acc - nn_test_acc) * 100.0 # in percentage points
if improvement >= 0:
  print(f"CNN performed better than NN by {improvement:.2f} percentage points.")
else:
  print(f"NN performed better than CNN by {-improvement:.2f} percentage points.")


# Plot training and validation accuracy curves for both models
plt.figure(figsize=(12,5))
# NN accuracy
plt.subplot(1,2,1)
plt.plot(history_nn.history['accuracy'], label='NN train acc')
plt.plot(history_nn.history['val_accuracy'], label='NN val acc')
plt.title('NN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


# CNN accuracy
plt.subplot(1,2,2)
plt.plot(history_cnn.history['accuracy'], label='CNN train acc')
plt.plot(history_cnn.history['val_accuracy'], label='CNN val acc')
plt.title('CNN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()


# Plot training and validation loss for both models
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history_nn.history['loss'], label='NN train loss')
plt.plot(history_nn.history['val_loss'], label='NN val loss')
plt.title('NN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


plt.subplot(1,2,2)
plt.plot(history_cnn.history['loss'], label='CNN train loss')
plt.plot(history_cnn.history['val_loss'], label='CNN val loss')
plt.title('CNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()