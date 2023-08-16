import tensorflow as tf

mobile = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False)