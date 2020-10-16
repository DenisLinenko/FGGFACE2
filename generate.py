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


# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
	# load image from file
	pixels = pyplot.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)

	print(results)

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
	
	
	# create predication model - note that we have several 
	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	# perform prediction
	yhat = model.predict(samples)

	return yhat

# define filenames
# t1_start = perf_counter()

try:
	
	embeddings = get_embeddings([sys.argv[1]])
	
	vector = embeddings[0]

	# serializing vector to string and saving to file

	memfile = io.BytesIO()
	np.save(memfile, vector)
	memfile.seek(0)

	serialized = json.dumps(memfile.read().decode('latin-1'))

	# writing file (we can  keen this info in DB as well)
	file = open(sys.argv[1]+'_vector', "w")
	n = file.write(serialized)
	file.close()
except Exception as err:
	print('Unable to generate vector, due to: ', err)