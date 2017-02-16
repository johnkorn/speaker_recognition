#!/usr/local/bin/python
#!/usr/bin/env PYTHONIOENCODING="utf-8" python
import os
import wave

import tflearn
from tflearn.data_utils import image_preloader
from tflearn.data_utils import build_hdf5_image_dataset
import h5py

from random import shuffle

import numpy

from googlenet import googlenet
from tflearn.layers.estimator import regression

import tensorflow as tf
print("You are using tensorflow version "+ tf.__version__) #+" tflearn version "+ tflearn.version)
if tf.__version__ >= '0.12' and os.name == 'nt':
	print("sorry, tflearn is not ported to tensorflow 0.12 on windows yet!(?)")
	quit() # why? works on Mac?



def dense_to_one_hot(batch, batch_size, num_labels):
	sparse_labels = tf.reshape(batch, [batch_size, 1])
	indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
	concatenated = tf.concat(1, [indices, sparse_labels])
	concat = tf.concat(0, [[batch_size], [num_labels]])
	output_shape = tf.reshape(concat, [2])
	sparse_to_dense = tf.sparse_to_dense(concatenated, output_shape, 1.0, 0.0)
	return tf.reshape(sparse_to_dense, [batch_size, num_labels])

def dense_to_one_hot(labels_dense, num_classes=10):
	"""Convert class labels from scalars to one-hot vectors."""
	return numpy.eye(num_classes)[labels_dense]

def load_wav_file(name):
	f = wave.open(name, "rb")
	# print("loading %s"%name)
	chunk = []
	data0 = f.readframes(CHUNK)
	while data0:  # f.getnframes()
		# data=numpy.fromstring(data0, dtype='float32')
		# data = numpy.fromstring(data0, dtype='uint16')
		data = numpy.fromstring(data0, dtype='uint8')
		data = (data + 128) / 255.  # 0-1 for Better convergence
		# chunks.append(data)
		chunk.extend(data)
		data0 = f.readframes(CHUNK)
	# finally trim:
	chunk = chunk[0:CHUNK * 2]  # should be enough for now -> cut
	chunk.extend(numpy.zeros(CHUNK * 2 - len(chunk)))  # fill with padding 0's
	# print("%s loaded"%name)
	return chunk


def get_libri_data(path):    
    def ispng(name):
        return "png"==name.split(".")[-1]

    labels=[]
    data=[]
    speakers = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
    for speaker in speakers:
            speaker_dir = os.path.join(path,speaker)
            subdirs = [os.path.join(speaker_dir,d) for d in os.listdir(speaker_dir) if os.path.isdir(os.path.join(speaker_dir,d))]
            for subdir in subdirs:
                # now get flac files and process them
                files = [os.path.join(subdir,f) for f in os.listdir(subdir) if ispng(os.path.join(subdir,f))]
                for file in files:    
                    labels.append(speaker)
                    data.append(file)
    #print("Got %s images" % len(labels))                
    return data, labels

TRAIN_DATA_PATH = "data/LibriSpeech/test-clean"
DATASET_FILE = "data/LibriSpeech/test-clean-spectro.txt"

def create_dataset_file():
    files, speakers= get_libri_data(TRAIN_DATA_PATH)
    sp_set = set(speakers)
    num_classes=len(sp_set)
    sp_list = list(sp_set)
    speakers_map = {}
    labels=[]
    for i in range(num_classes):
        speakers_map[sp_list[i]]=i
    
    f = open(DATASET_FILE, 'w')
    lines = []
    for fname, speaker in zip(files,speakers):
        label = speakers_map[speaker]      
        lines.append("%s %s\n" % (os.path.abspath(fname),label))
    shuffle(lines)
    f.writelines(lines)
    f.close()
    
    f = open("test_speakers_map.txt", 'w')
    lines=[]
    lines.append("index speaker_id\n")
    for k in speakers_map:
        lines.append("%d %d\n" % (speakers_map[k], int(k)))
    f.writelines(lines)
    f.close()
    
    return len(files), num_classes, speakers_map


#if not os.path.isfile(DATASET_FILE):
#	create_dataset_file()
num_files, num_classes, sp_map = create_dataset_file()
print "Got %d files of %d speakers" % (num_files, num_classes)

input_width = 227

def create_hdf5_dataset(dataset_file, output_name):
	print('Building HDF5 dataset to %s ...' % output_name)
	build_hdf5_image_dataset(dataset_file, image_shape=(input_width, input_width), mode='file', 
		output_path=output_name, categorical_labels=True, normalize=True)


def get_data_hdf5(dataset_file):	
	h5f = h5py.File(dataset_file, 'r')
	X = h5f['X']
	Y = h5f['Y']
	return X, Y

def get_data_preloader():
	X, Y = image_preloader(DATASET_FILE, image_shape=(input_width, input_width), mode='file', 
                       categorical_labels=True, normalize=True, grayscale=False)
	return X, Y

h5_filename = 'data/LibriSpeech/testdata.h5'
if not os.path.isfile(h5_filename):
	create_hdf5_dataset(DATASET_FILE, h5_filename)
X, Y = get_data_hdf5(h5_filename)


'''
To make GPU invisible:
export CUDA_VISIBLE_DEVICES=""

To return to normal:
unset CUDA_VISIBLE_DEVICES
'''

BATCH_SIZE = 32

tf.reset_default_graph()
tflearn.init_graph(num_cores=2, gpu_memory_fraction=0.5)

loss = googlenet(input_width, num_classes)
network = tflearn.regression(loss, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
model = tflearn.DNN(network, checkpoint_path='checkpoints/model_googlenet',
                    max_checkpoints=1, tensorboard_verbose=2)


model.load('checkpoints/model_googlenet-10350')

model.fit(X, Y, n_epoch=200, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=BATCH_SIZE, snapshot_step=150,
          snapshot_epoch=False, run_id='speakers_googlenet_ontestset')

model.save('speakers_googlenet_on_testset.tflearn')