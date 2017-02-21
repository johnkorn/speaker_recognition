import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.externals import joblib

from scipy.misc import imread, imresize
import numpy  as np

import tensorflow as tf
import tflearn

from googlenet import googlenet_core as feature_layer
from googlenet import googlenet


class SpeakerVerifier(object):

	def __init__(self):		
		self.TRAINDATA_PATH = 'recordings_train/'
		self.EVAL_PATH = 'recordings_eval/'
		self.DICT_NAME = 'users'
		self.dict_path = self.TRAINDATA_PATH+self.DICT_NAME+'.txt'
		self.sep = '\t'

		self.FEATURES_PATH = self.TRAINDATA_PATH + 'features.txt'
		self.clf_path = self.TRAINDATA_PATH + 'knn.pkl'

		# kNN model params
		self.n_neighbors = 5
		self.weights = 'distance' #'uniform'


		self.MODEL_PATH = 'checkpoints/model_googlenet-12000'
		self.input_width = 227
		self.num_classes = 40

		self.users = self.read_dict()
		self.model = self.load_model()


	def load_model(self):
		tf.reset_default_graph()
		tflearn.init_graph(num_cores=2, gpu_memory_fraction=0.5)

		features = feature_layer(self.input_width)
		loss = tflearn.layers.core.fully_connected(features, self.num_classes, activation='softmax')
		network = tflearn.regression(loss, optimizer='momentum',
					loss='categorical_crossentropy',
					learning_rate=0.001)
		model = tflearn.DNN(network, checkpoint_path='checkpoints/model_googlenet',
					max_checkpoints=1, tensorboard_verbose=2)


		model.load(self.MODEL_PATH, weights_only=True)
		m = tflearn.DNN(features, session=model.session)

		return m

	def read_dict(self):
			if os.path.isfile(self.dict_path):
				with open(self.dict_path, "r") as f:
					dict = {}
					for line in f:
						values = line.split(self.sep)
						dict[int(values[0])] = values[1].strip()
					return(dict)
			else:
				# return empty dictionary
				return {}

	def write_dict(dict, filename, sep):
		with open(filename, "w") as f:
			for i in dict.keys():            
				f.write(str(i) + sep + dict[i] + "\n")
	

	def wav_to_flac(filename):
		wavdata, samplerate = sf.read(filename)
		sf.write(filename + '.flac', wavdata, samplerate)
	
		return filename + '.flac'

	
	def get_audio_info(audio_file):
		""" Reads flac files
		"""
		samples, samplerate = sf.read(audio_file)
		return samples, samplerate


	def graph_spectrogram(audio_file, img_path):
		sound_info, frame_rate = get_audio_info(audio_file)
		plt.figure(num=None, figsize=(19, 12), frameon=False)
	
		fig,ax = plt.subplots(1)
		# Set whitespace to 0
		fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
	
	
		plt.specgram(sound_info, Fs=frame_rate)
	
		plt.axis('tight')
		plt.axis('off')
	
		plt.savefig(img_path) #bbox_inches='tight', pad_inches=0)  
		plt.clf()
		plt.close()

	
	def get_features(input_path):
		""" Calculates activations of hidden layer of trained NN"""

		img= imread(input_path, mode='RGB')
		img = imresize(img, (self.input_width, self.input_width))


		res = np.array(self.model.predict([img]))
		return res.flatten()


	def extract_feaures_for_dir(dirname, label=None):
	
		def ispng(name):
			return "png"==name.split(".")[-1]
		
		files = [os.path.join(dirname,f) for f in os.listdir(dirname) if ispng(os.path.join(dirname,f))]
		all_features = []

		for file in files:
			#print('extracting features from ' + file + ' ...')
			f = get_features(self.model, file)		
			if label is not None:
				f = np.concatenate([[label], f])
			all_features.append(f)

		return np.array(all_features, dtype=np.float32)


	def get_xy_pairs(features):
		y = features[:,0]
		X = features[:,1:]
		return X, y


	def training_routine(files, label):
		# calculate feature vectors           !!!!!!!!!!!!
		feature_set = extract_feaures_for_dir(self.TRAINDATA_PATH, label)

		
		# append to existing features
		if os.path.isfile(self.FEATURES_PATH):
			data = np.loadtxt(self.FEATURES_PATH, dtype=np.float32)
			feature_set = np.concatenate([data, feature_set], axis=0)		

		
		# FIT kNN MODEL
		clf = neighbors.KNeighborsClassifier(self.n_neighbors, weights=self.weights)
		#clf = neighbors.RadiusNeighborsClassifier(radius = 1.0, weights=weights)
		X, y = get_xy_pairs(feature_set)
		clf.fit(X, y)
		joblib.dump(clf, self.clf_path) 


		# saving features
		with open(self.FEATURES_PATH,'w') as f:
			np.savetxt(f, feature_set, fmt='%.10f')
	

	def train_existing_user(files, user_id):
		if len(self.users)==0:
			return "List of Users is empty!"

		keys = [int(k) for k in self.users.keys()]
		if not user_id in keys:
			return "Invalid user ID: " % str(user_id)
			
		training_routine(files, user_id)
		

	def train_new_user(files, user_name):		
		label = 1
		if len(self.users)>0:
			keys = [int(k) for k in self.users.keys()]
			key = np.max(keys)+1
			users[key]=name
			global label
			label = key
		else:
			self.users = {1: user_name.strip()}


		training_routine(files, label)

		# save dictionary 
		if len(self.users.keys())>count:
			write_dict(self.users, self.dict_path, sep)


	def verification(files):
		features = extract_feaures_for_dir(self.EVAL_PATH, None)
		clf = joblib.load(self.clf_path)
		for i in range(len(features)):
			answer = int(clf.predict(features[i].reshape(1,-1))[0])
			dist, ind = clf.kneighbors(features[i].reshape(1,-1))
			if np.min(dist)<900.0:
				#print('Prediction is: %s (%s)' % (answer, users[answer]))
				return 'Prediction is: %s (%s)' % (answer, self.users[answer])
			else:
				#print('Sample is unknown!')
				return 'Sample is unknown!'


	

