import os
import time
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import Callback
from typing import Tuple, Any
from tensorflow.keras.applications import VGG16

# Define the MullerResizer class
class MullerResizer(tf.keras.layers.Layer):
    """Learned Laplacian resizer in Keras Layer."""

    def __init__(
        self,
        target_size=(224, 224),
        base_resize_method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
        kernel_size=5,
        stddev=1.0,
        num_layers=2,
        avg_pool=False,
        dtype=tf.float32,
        init_weights=None,
        name='muller_resizer',
    ):
        super().__init__(name=name)
        self._target_size = target_size
        self._base_resize_method = base_resize_method
        self._antialias = antialias
        self._kernel_size = kernel_size
        self._stddev = stddev
        self._num_layers = num_layers
        self._avg_pool = avg_pool
        self._dtype = dtype
        self._init_weights = init_weights

    def build(self, input_shape):
        self._weights = []
        self._biases = []
        for layer in range(1, self._num_layers + 1):
            weight = self.add_weight(
                name='weight_' + str(layer),
                shape=[],
                dtype=self._dtype,
                initializer=tf.keras.initializers.Constant(
                    self._init_weights[2 * layer - 2]
                ) if self._init_weights else tf.keras.initializers.zeros(),
            )
            bias = self.add_weight(
                name='bias_' + str(layer),
                shape=[],
                dtype=self._dtype,
                initializer=tf.keras.initializers.Constant(
                    self._init_weights[2 * layer - 1]
                ) if self._init_weights else tf.keras.initializers.zeros(),
            )
            self._weights.append(weight)
            self._biases.append(bias)

        super().build(input_shape)

    def _base_resizer(self, inputs):
        """Base resizer function for muller."""
        stride = [
            1,
            inputs.get_shape().as_list()[1] // self._target_size[0],
            inputs.get_shape().as_list()[2] // self._target_size[1],
            1
        ]
        if self._avg_pool and stride[1] > 1 and stride[2] > 1:
            pooling_shape = [1, stride[1], stride[2], 1]
            inputs = tf.nn.avg_pool(inputs, pooling_shape, stride, padding='SAME')

        return tf.cast(
            tf.image.resize(
                inputs,
                self._target_size,
                method=self._base_resize_method,
                antialias=self._antialias),
            self._dtype)

    def _gaussian_blur(self, inputs):
        """Gaussian blur function for muller."""
        stddev = tf.cast(self._stddev, self._dtype)
        size = self._kernel_size
        radius = size // 2
        x = tf.cast(tf.range(-radius, radius + 1), self._dtype)
        blur_filter = tf.exp(-tf.pow(x, 2.0) / (2.0 * tf.pow(stddev, 2.0)))
        blur_filter /= tf.reduce_sum(blur_filter)
        blur_v = tf.reshape(blur_filter, [size, 1, 1, 1])
        blur_h = tf.reshape(blur_filter, [1, size, 1, 1])
        num_channels = inputs.get_shape()[-1]
        blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
        blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
        blurred = tf.nn.depthwise_conv2d(
            inputs, blur_h, strides=[1, 1, 1, 1], padding='SAME')
        blurred = tf.nn.depthwise_conv2d(
            blurred, blur_v, strides=[1, 1, 1, 1], padding='SAME')
        return blurred

    def call(self, inputs):
        inputs.get_shape().assert_has_rank(4)
        if inputs.dtype != self._dtype:
            inputs = tf.cast(inputs, self._dtype)

        # Creates the base resized image.
        net = self._base_resizer(inputs)

        # Multi Laplacian resizer.
        for weight, bias in zip(self._weights, self._biases):
            blurred = self._gaussian_blur(inputs)
            residual_image = blurred - inputs
            resized_residual = self._base_resizer(residual_image)
            net = net + tf.nn.tanh(weight * resized_residual + bias)
            inputs = blurred
        return net

# Function to read filenames and labels from a file
def read_split_file(split_file):
    images = []
    labels = []
    with open(split_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                images.append(parts[0])
                labels.append(int(parts[1]))
    return images, labels

# Function to prepare data directories from split files
def prepare_data_from_split_file(split_file, src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    images, labels = read_split_file(split_file)
    for img, lbl in zip(images, labels):
        class_dir = os.path.join(dest_dir, str(lbl))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        src_path = os.path.join(src_dir, img)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(class_dir, img))

# Prepare training and testing datasets
train_split_file = 'train_split.txt'
test_split_file = 'test_split.txt'
src_dir = 'images'  # Path to your images directory
train_dest_dir = 'insectTrainCNN_prepared'
test_dest_dir = 'insectTestCNN_prepared'

prepare_data_from_split_file(train_split_file, src_dir, train_dest_dir)
prepare_data_from_split_file(test_split_file, src_dir, test_dest_dir)

# Create ImageDataGenerators
with tf.device('/CPU:0'):
    # Create ImageDataGenerators
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dest_dir,
        target_size=(256, 256),
        batch_size=1,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dest_dir,
        target_size=(256, 256),
        batch_size=1,
        class_mode='categorical'
    )

# Print class indices to debug the class mismatch issue
print("Training classes:", train_generator.class_indices)
print("Testing classes:", test_generator.class_indices)

# Define the model
def build_model(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    base_model.trainable = False  # Freeze the base model

    # Gradually increasing the number of neurons in each dense layer with batch normalization
    model = models.Sequential()
    model.add(MullerResizer(target_size=(256, 256), num_layers=3, name='muller_resizer'))
    model.add(base_model)
    model.add(layers.Conv2D(512, (3, 3), activation='tanh', padding='same'))  # Added Conv2D layer
    model.add(layers.MaxPooling2D((2, 2)))  # Added MaxPooling2D layer
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='tanh'))
    model.add(layers.Dense(4096, activation='tanh'))
    model.add(layers.Dropout(0.2))

    # Final output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    #SGD = SGD(learning_rate=1e-5, momentum=0.9)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Reduce learning rate
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
# Custom callback to stop training if the performance at each 20th epoch is not the highest in the last 5 epochs
class CustomEarlyStoppingAtIntervals(Callback):
    def __init__(self, monitor='val_accuracy', interval=400, lookback=5):
        super(CustomEarlyStoppingAtIntervals, self).__init__()
        self.monitor = monitor
        self.interval = interval
        self.lookback = lookback
        self.best = -float('inf')
        self.recent_best = -float('inf')
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        self.history.append(current)
        if current > self.best:
            self.best = current

        if len(self.history) > self.lookback:
            self.history.pop(0)

        if (epoch + 1) % self.interval == 0:
            self.recent_best = max(self.history)
            if current < self.recent_best:
                self.model.stop_training = True
                print(f"Stopping training at epoch {epoch + 1} as {self.monitor} did not improve from the best {self.recent_best} in the last {self.lookback} epochs")

# Define callbacks for training
class SleepAfterEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, sleep_time=30):
        super(SleepAfterEpochCallback, self).__init__()
        self.sleep_time = sleep_time

    def on_epoch_end(self, epoch, logs=None):
        print(f"Sleeping for {self.sleep_time} seconds after epoch {epoch + 1}")
        time.sleep(self.sleep_time)

# Define callbacks for training
sleep_callback = SleepAfterEpochCallback(sleep_time=60)
final_callbacks = [
    CustomEarlyStoppingAtIntervals(monitor='val_accuracy', interval=400, lookback=5),
    sleep_callback
]


# Determine the number of classes in training and testing datasets
num_classes_train = len(train_generator.class_indices)
num_classes_test = len(test_generator.class_indices)

# Build and train the model
model = build_model(num_classes_train)

history = model.fit(train_generator, epochs=100, validation_data=test_generator, callbacks=final_callbacks)

# Save the model weights
model.save_weights('cnn_dg_weights.weights.h5')

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}")
