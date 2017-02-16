from scipy.misc import imread, imresize
import tflearn
import numpy  as np

from sklearn import neighbors
from sklearn.externals import joblib

import os
import sys

from googlenet import googlenet_core as feature_layer
from googlenet import googlenet

import tensorflow as tf

MODEL_PATH = 'checkpoints/model_googlenet-12000'
input_width = 227
num_classes = 40

def load_model():
    tf.reset_default_graph()
    tflearn.init_graph(num_cores=2, gpu_memory_fraction=0.5)

    features = feature_layer(input_width)
    loss = tflearn.layers.core.fully_connected(features, num_classes, activation='softmax')
    network = tflearn.regression(loss, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
    model = tflearn.DNN(network, checkpoint_path='checkpoints/model_googlenet',
                    max_checkpoints=1, tensorboard_verbose=2)


    model.load(MODEL_PATH, weights_only=True)
    
    
    m = tflearn.DNN(features, session=model.session)
    
    return m



def get_features(model, input_path):
    """ Calculates activations of hidden layer of trained NN
    """
    
    img= imread(input_path, mode='RGB')
    img = imresize(img, (input_width, input_width))


    res = np.array(model.predict([img]))
    return res.flatten()


def extract_feaures_for_dir(dirname, label=None):
	
	def ispng(name):
		return "png"==name.split(".")[-1]

	model = load_model()
	files = [os.path.join(dirname,f) for f in os.listdir(dirname) if ispng(os.path.join(dirname,f))]
	all_features = []

	for file in files:
		print('extracting features from ' + file + ' ...')
		f = get_features(model, file)		
		if label is not None:
			f = np.concatenate([[label], f])
		all_features.append(f)

	return np.array(all_features, dtype=np.float32)



def write_dict(dict, filename, sep):
    with open(filename, "w") as f:
        for i in dict.keys():            
            f.write(str(i) + sep + dict[i] + "\n")

def read_dict(filename, sep):
    with open(filename, "r") as f:
        dict = {}
        for line in f:
            values = line.split(sep)
            dict[int(values[0])] = values[1].strip()
        return(dict)


TRAINDATA_PATH = 'recordings_train/'
EVAL_PATH = 'recordings_eval/'
DICT_NAME = 'users'
dict_path = TRAINDATA_PATH+DICT_NAME+'.txt'
sep = '\t'

FEATURES_PATH = TRAINDATA_PATH + 'features.txt'
clf_path = TRAINDATA_PATH + 'knn.pkl'

# kNN model params
n_neighbors = 5
weights = 'distance' #'uniform'

def get_xy_pairs(features):
	y = features[:,0]
	X = features[:,1:]
	return X, y

def run_training():		
	# read dictionary and print existing users
	users = {}
	count = 0
	
	label=0
	if os.path.isfile(dict_path):		
		users = read_dict(dict_path, sep)		
		keys = [int(k) for k in users.keys()]
		count = len(keys)

		print 'Select existing user or add a new one:'
		for k in users:
			print '%s:\t%s' % (str(k), users[k])
		print '0:\tADD NEW USER'		
		choice = int(raw_input('\nType in selected index: ').strip())
		if choice==0:
			name = raw_input("Type new user's name: ").strip()
			key = np.max(keys)+1
			users[key]=name
			label=key
		else: 
			label = choice
	else:
		# add label (user name) to empty dictionary	
		name = raw_input("Type new user's name: ")
		users = {1: name.strip()}
		label = 1


	# calculate feature vectors 
	feature_set = extract_feaures_for_dir(TRAINDATA_PATH, label)

	# save dictionary and features
	if len(users.keys())>count:
		write_dict(users, dict_path, sep)
	
	# append to existing features
	if os.path.isfile(FEATURES_PATH):
		data = np.loadtxt(FEATURES_PATH, dtype=np.float32)
		feature_set = np.concatenate([data, feature_set], axis=0)		

	print('fitting kNN model...')
	clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
	#clf = neighbors.RadiusNeighborsClassifier(radius = 1.0, weights=weights)
	X, y = get_xy_pairs(feature_set)
	clf.fit(X, y)
	joblib.dump(clf, clf_path) 


	print('saving features...')
	with open(FEATURES_PATH,'w') as f:
		np.savetxt(f, feature_set, fmt='%.10f')
	
	print('Done training!')


def run_eval():
	users = read_dict(dict_path, sep)
	print(users)
	features = extract_feaures_for_dir(EVAL_PATH, None)
	clf = joblib.load(clf_path)
	for i in range(len(features)):
		answer = int(clf.predict(features[i].reshape(1,-1))[0])
		dist, ind = clf.kneighbors(features[i].reshape(1,-1))
		if np.min(dist)<900.0:
			print('Prediction is: %s (%s)' % (answer, users[answer]))
		else:
			print('Sample is unknown!')
		#print ("Distances: ")
		#print dist
		#print ("Indicies: ")
		#print ind


def dialog():
	choice = raw_input('''Available modes:
		1: training
		2: evaluation
		Your choice: ''')
	choice = int(choice)
	if choice==1:
		run_training()
	else:
		run_eval()

def init_paths():
	DICT_NAME = 'users'
	dict_path = TRAINDATA_PATH+DICT_NAME+'.txt'
	sep = '\t'

	FEATURES_PATH = TRAINDATA_PATH + 'features.txt'
	clf_path = TRAINDATA_PATH + 'knn.pkl'


#res = np.array(get_features('recordings_train/84-121550-0003.png'))
if(__name__ == '__main__'):
	
	if len(sys.argv)==2:
		arg = sys.argv[1]
		EVAL_PATH = arg
	elif len(sys.argv)==3:
		TRAINDATA_PATH = sys.argv[1]
		EVAL_PATH = sys.argv[2]
		dict_path = TRAINDATA_PATH+DICT_NAME+'.txt'
		sep = '\t'

		FEATURES_PATH = TRAINDATA_PATH + 'features.txt'
		clf_path = TRAINDATA_PATH + 'knn.pkl'
	
	init_paths()
	print TRAINDATA_PATH
	print EVAL_PATH
	print dict_path
	print clf_path
	print FEATURES_PATH

	dialog()