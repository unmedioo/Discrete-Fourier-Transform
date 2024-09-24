import os
import numpy as np
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# 文件路径
train_dir = 'C:\\Users\\admin\\Pictures\\MNIST\\mnist_train'
test_dir = 'C:\\Users\\admin\\Pictures\\MNIST\\mnist_test'
model_path = 'C:\\Users\\admin\\PycharmProjects\\sztxcl\\my_model.h5'

def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
                img_array = img_to_array(img)
                # 应用傅里叶变换
                f_transform = np.fft.fft2(img_array)
                f_transform_shifted = np.fft.fftshift(f_transform)
                magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1e-8)
                images.append(magnitude_spectrum)
                labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# 加载数据
x_train, y_train = load_data(train_dir)
x_test, y_test = load_data(test_dir)

# 规范化数据
x_train = x_train / 255.0
x_test = x_test / 255.0

# 扩展数据维度以适应VGG16输入要求
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
x_train = np.repeat(x_train, 3, axis=-1)
x_test = np.repeat(x_test, 3, axis=-1)

# 转换标签为独热编码
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 使用VGG16模型
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 使用数据增强
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# 训练模型
model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=10, validation_data=(x_test, y_test))

# 保存模型
model.save(model_path)
