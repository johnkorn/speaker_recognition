import tensorflow as tf
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
import os

from tflearn.data_utils import build_hdf5_image_dataset
import h5py

from random import shuffle

import numpy


def vgg16(input, num_class):

    x = tflearn.conv_2d(input, 64, 3, activation='relu', scope='conv1_1')
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
    x = tflearn.dropout(x, 0.5, name='dropout2')

    x = tflearn.fully_connected(x, num_class, activation='softmax', scope='fc8',
                                restore=False)

    return x


TRAIN_DATA_PATH = "data/LibriSpeech/test-clean"
DATASET_FILE = "data/LibriSpeech/test-clean-spectro_vgg.txt"

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
    
    f = open("test_speakers_map_vgg.txt", 'w')
    lines=[]
    lines.append("index speaker_id\n")
    for k in speakers_map:
        lines.append("%d %d\n" % (speakers_map[k], int(k)))
    f.writelines(lines)
    f.close()
    
    return len(files), num_classes, speakers_map

num_files, num_classes, sp_map = create_dataset_file()
print "Got %d files of %d speakers" % (num_files, num_classes)


model_path = "models/"
input_width = 224

def create_hdf5_dataset(dataset_file, output_name):
    print('Building HDF5 dataset to %s ...' % output_name)
    build_hdf5_image_dataset(dataset_file, image_shape=(input_width, input_width), mode='file', 
        output_path=output_name, categorical_labels=True, normalize=False)

def get_data_hdf5(dataset_file):    
    h5f = h5py.File(dataset_file, 'r')
    X = h5f['X']
    Y = h5f['Y']
    return X, Y

h5_filename = 'data/LibriSpeech/testdata_vgg.h5'
if not os.path.isfile(h5_filename):
    create_hdf5_dataset(DATASET_FILE, h5_filename)
X, Y = get_data_hdf5(h5_filename)


tf.reset_default_graph()
tflearn.init_graph(num_cores=2, gpu_memory_fraction=0.7)

# VGG preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center(mean=[123.68, 116.779, 103.939],
                                     per_channel=True)
# VGG Network
x = tflearn.input_data(shape=[None, input_width, input_width, 3], name='input',
                       data_preprocessing=img_prep)
softmax = vgg16(x, num_classes)
regression = tflearn.regression(softmax, optimizer='adam',
                                loss='categorical_crossentropy',
                                learning_rate=0.001, restore=False)

model = tflearn.DNN(regression, checkpoint_path='checkpoints/model_vgg_finetuning',
                    max_checkpoints=1, tensorboard_verbose=2)

model_file = os.path.join(model_path, "vgg16.tflearn")
model.load(model_file, weights_only=True)

# Start finetuning
model.fit(X, Y, n_epoch=10, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=16, snapshot_epoch=True,
          run_id='vgg-finetuning')

model.save('speakers_testset_retrained_vgg.tflearn')