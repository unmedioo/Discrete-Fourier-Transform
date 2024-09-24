import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
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
                img = load_img(img_path, color_mode='grayscale')
                img_array = img_to_array(img)
                # 应用傅里叶变换
                f_transform = np.fft.fft2(img_array)
                f_transform_shifted = np.fft.fftshift(f_transform)
                magnitude_spectrum = 20*np.log(np.abs(f_transform_shifted))
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

# 转换标签为独热编码
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 构建模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# 保存模型
model.save(model_path)
