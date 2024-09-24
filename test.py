import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# 归一化数据
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 对训练集和测试集进行傅里叶变换
def apply_fft(images):
    fft_images = np.zeros_like(images)
    for i in range(len(images)):
        img = images[i]
        fft_img = np.fft.fft2(img)
        fft_shifted = np.fft.fftshift(fft_img)
        fft_images[i] = np.abs(fft_shifted)
    return fft_images

x_train_fft = apply_fft(x_train)
x_test_fft = apply_fft(x_test)

# 转换标签为one-hot编码
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 定义模型
model = Sequential()
input_layer = Input(shape=input_shape)  # 使用Input对象定义输入层
x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(10, activation='softmax')(x)

# 创建模型
model = Model(inputs=input_layer, outputs=outputs)

# 编译模型
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # 使用learning_rate参数
              metrics=['accuracy'])

# 自定义回调函数来计算每个类别的准确率
class ClassAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        super().__init__()
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        x_data, y_data = self.test_data
        predictions = self.model.predict(x_data)
        correct_predictions = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_data, axis=1))
        total_samples = len(y_data)
        class_accuracies = {}
        for i in range(10):  # 假设有10个类别
            class_mask = np.where(np.argmax(y_data, axis=1) == i)[0]
            class_predictions = predictions[class_mask]
            class_true_labels = y_data[class_mask]
            class_correct = np.sum(np.argmax(class_predictions, axis=1) == np.argmax(class_true_labels, axis=1))
            class_accuracies[f'class_{i}'] = class_correct / len(class_mask)
        print(f'Class accuracies: {class_accuracies}')

# 配置回调函数
callbacks = [ClassAccuracyCallback((x_test_fft, y_test))]

# 训练模型并记录历史
history = model.fit(x_train_fft, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(x_test_fft, y_test), callbacks=callbacks)

# 可视化训练过程
plt.figure(figsize=(12, 4))

# 绘制训练集的损失和准确率
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制验证集的损失和准确率
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
