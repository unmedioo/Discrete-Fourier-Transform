import numpy as np
import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from scipy.fftpack import fft2, ifft2


# 定义傅里叶变换函数
def fourier_transform(image):
    return np.abs(fft2(image))


# 加载MNIST数据集
def load_mnist_data(train_dir, test_dir):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # 遍历训练集
    for label in os.listdir(train_dir):
        label_dir = os.path.join(train_dir, label)
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            image = Image.open(image_path).convert('L')
            image_array = np.array(image)
            transformed_image = fourier_transform(image_array)
            X_train.append(transformed_image)
            y_train.append(int(label))

    # 遍历测试集
    for label in os.listdir(test_dir):
        label_dir = os.path.join(test_dir, label)
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            image = Image.open(image_path).convert('L')
            image_array = np.array(image)
            transformed_image = fourier_transform(image_array)
            X_test.append(transformed_image)
            y_test.append(int(label))

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


# 应用傅里叶变换并加载数据
X_train, y_train, X_test, y_test = load_mnist_data(train_dir='C:\\Users\\admin\\Pictures\\MNIST\\mnist_train',
                                                   test_dir='C:\\Users\\admin\\Pictures\\MNIST\\mnist_test')

# 将标签转换为one-hot编码
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=10)

# 定义模型
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train_one_hot, epochs=10, batch_size=32, validation_data=(X_test, y_test_one_hot))

# 保存模型
model_path = 'C:\\Users\\admin\\PycharmProjects\\sztxcl\\my_model.h5'
model.save(model_path)
