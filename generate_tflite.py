
# face verification with the VGGFace2 model
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from time import perf_counter
import sys
import numpy as np
import io
import json
import tensorflow as tf
from keras.models import load_model

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
	# load image from file
	pixels = pyplot.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)

	print("-------results----------->", results)

	print(len(results), 'faces have been found')

	#--------------------------- detecting the number of faces ------------------------------------
	if len(results) == 0:
		raise Exception('No faces detected')
	elif len(results) > 1:
		raise Exception('Multiple faces detected')		

	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)

	return face_array

# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
	# extract faces
	faces = [extract_face(f) for f in filenames]
	# convert into an array of samples
	samples = asarray(faces, 'float32')
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input(samples, version=2)
	
	# Load the TFLite model and allocate tensors. View details
	interpreter = tf.lite.Interpreter(model_path="model_resnet50.tflite")
	interpreter.allocate_tensors()

	# Get input and output tensors.
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	# Test the model on input data.
	# input_shape = input_details[0]['shape']
	
	# Use same image as Keras model
	input_data = np.array(samples, dtype=np.float32)
	interpreter.set_tensor(input_details[0]['index'], input_data)

	interpreter.invoke()

	# The function `get_tensor()` returns a copy of the tensor data.
	# Use `tensor()` in order to get a pointer to the tensor.
	output_data = interpreter.get_tensor(output_details[0]['index'])
	
	return output_data

try:
	
	embeddings = get_embeddings([sys.argv[1]])
	
	vector = embeddings[0]

	# serializing vector to string and saving to file

	memfile = io.BytesIO()
	np.save(memfile, vector)
	memfile.seek(0)

	serialized = json.dumps(memfile.read().decode('latin-1'))

	# writing file (we can  keen this info in DB as well)
	file = open(sys.argv[1]+'_vector_l', "w")
	n = file.write(serialized)
	file.close()
except Exception as err:
	print('Unable to generate vector, due to: ', err)