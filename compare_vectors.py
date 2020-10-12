import sys
import json
import io
import numpy as np
from scipy.spatial.distance import cosine

def getEmbedding(filename):
	#  Just returning embedding from file (we can get info from DB as well)
	f = open(filename, "r").read()
	memfile = io.BytesIO()
	memfile.write(json.loads(f).encode('latin-1'))
	memfile.seek(0)
	return np.load(memfile)

def is_match(known_embedding, candidate_embedding, thresh=0.5):
	# calculate distance between embeddings
	score = cosine(known_embedding, candidate_embedding)
	if score <= thresh:
		print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
	else:
		print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))


print('comparing', sys.argv[1], sys.argv[2])

embedding1 = getEmbedding(sys.argv[1])
embedding2 = getEmbedding(sys.argv[2])

is_match(embedding1, embedding2)
