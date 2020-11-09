from numpy import asarray
from mtcnn.mtcnn import MTCNN
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import tensorflow as tf

#------------------resnet50 to tflite_model----------------


# model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# print('loading...')

# converter = tf.lite.TFLiteConverter.from_keras_model(model)

# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# print('converting...')
# tflite_model = converter.convert()

# print('saving...')

# with open('model_resnet50.tflite', 'wb') as f:
# 	f.write(tflite_model)

# print('done...')

#------------------vgg16 to tflite_model----------------


model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')

print('loading...')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

print('converting...')
tflite_model = converter.convert()

print('saving...')

with open('model_resnet50.tflite', 'wb') as f:
	f.write(tflite_model)

print('done...')


#------------------resnet50 to tflite_model_fl16----------------

# model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# converter.target_spec.supported_types =  [tf.float16]

# tflite_model_fl16 = converter.convert()

# with open('rf16facenet.tflite', 'wb') as f:
# 	f.write(tflite_model_fl16)

# print('done...')

