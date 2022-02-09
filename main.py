import tensorflow as tf
from tensorflow.keras.models import load_model
model = load_model('./model')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with tf.io.gfile.GFile('hand_detector.tflite', 'wb') as f:
        f.write(tflite_model)