import cv2
import numpy as np
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense



def extract_features(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 计算傅里叶变换
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)

    # 计算功率谱密度
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift))

    # 对非常小的值进行置零处理，以避免在计算log时出现除以零的情况
    magnitude_spectrum[magnitude_spectrum < 1e-10] = 0

    # 返回功率谱密度作为特征
    return magnitude_spectrum


# 数据集路径
train_dir = 'C:\\Users\\admin\\Pictures\\MNIST\\mnist_train'
test_dir = 'C:\\Users\\admin\\Pictures\\MNIST\\mnist_test'

# 初始化特征和标签列表
features = []
labels = []

# 遍历训练集
for label in os.listdir(train_dir):
    label_path = os.path.join(train_dir, label)
    for filename in os.listdir(label_path):
        filepath = os.path.join(label_path, filename)
        feature = extract_features(filepath)
        features.append(feature)
        labels.append(int(label))  # 标签是数字0到9

# 遍历测试集
for label in os.listdir(test_dir):
    label_path = os.path.join(test_dir, label)
    for filename in os.listdir(label_path):
        filepath = os.path.join(label_path, filename)
        feature = extract_features(filepath)
        features.append(feature)
        labels.append(int(label))  # 标签是数字0到9

# 将数据转换为NumPy数组
features = np.array(features)
labels = np.array(labels)

# 将标签转换为one-hot编码
labels = to_categorical(labels)

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 构建模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # 输入图像尺寸为28x28像素
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 输出层有10个节点，对应于0到9的10个数字
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

model_path = 'C:\\Users\\admin\\PycharmProjects\\sztxcl\\my_model.h5'  # 这是模型的保存路径
model.save(model_path)