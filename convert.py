from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import tensorflow as tf

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

print('loading...')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print('converting...')
tflite_quant_model = converter.convert()

print('saving...')
with open('model.tflite', 'wb') as f:
	f.write(tflite_quant_model)

print('done...')


