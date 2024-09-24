import tensorflow as tf

tensorflow_version = tf.__version__
gpu_avilable = tf.test.is_gpu_available()

print("tensorflow version: ", tensorflow_version,"\tGPU aviable:", gpu_avilable)

a = tf.constant([1.0,2.0], name = 'a')
b = tf.constant([1.0,2.0], name = 'b')
result = tf.add(a,b,name='add')
print(result)

