# import tensorflow
import tensorflow as tf

hello = tf.constant("Hello, TensorFlow!")

# Create session
sess = tf.Session()

# run the operation
print(sess.run(hello))