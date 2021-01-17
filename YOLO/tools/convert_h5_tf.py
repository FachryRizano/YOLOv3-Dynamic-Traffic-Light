import tensorflow as tf
model = tf.keras.models.load_model('yolov3_custom_Tiny')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)