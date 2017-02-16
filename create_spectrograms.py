import numpy as np
import soundfile as sf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from numpy.lib import stride_tricks
import PIL.Image as Image
import os


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

# for flac files:
def get_audio_info(audio_file):
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


""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.rfft(frames)    
    
""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20., alpha=1.0, f0=0.9, fmax=1):
    spec = spec[:, 0:256]
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins) #** factor
    
    # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
    scale = np.array(map(lambda x: x * alpha if x <= f0 else (fmax-alpha*f0)/(fmax-f0)*(x-f0)+alpha*f0, scale))
    scale *= (freqbins-1)/max(scale)

    newspec = np.complex128(np.zeros([timebins, freqbins]))
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = [0.0 for i in range(freqbins)]
    totw = [0.0 for i in range(freqbins)]
    for i in range(0, freqbins):
        if (i < 1 or i + 1 >= freqbins):
            newspec[:, i] += spec[:, i]
            freqs[i] += allfreqs[i]
            totw[i] += 1.0
            continue
        else:
            # scale[15] = 17.2
            w_up = scale[i] - np.floor(scale[i])
            w_down = 1 - w_up
            j = int(np.floor(scale[i]))
           
            newspec[:, j] += w_down * spec[:, i]
            freqs[j] += w_down * allfreqs[i]
            totw[j] += w_down
            
            newspec[:, j + 1] += w_up * spec[:, i]
            freqs[j + 1] += w_up * allfreqs[i]
            totw[j + 1] += w_up
    
    for i in range(len(freqs)):
        if (totw[i] > 1e-6):
            freqs[i] /= totw[i]
    
    return newspec, freqs

### Block Processing
### Sound files can also be read in short, optionally overlapping blocks with 
### soundfile.blocks(). For example, this calculates the signal level for each 
### block of a long file:
# rms = [np.sqrt(np.mean(block**2)) for block in
# sf.blocks('myfile.wav', blocksize=1024, overlap=512)]

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="gray", channel=0, name='tmp.png', alpha=1, offset=0):
    samples, samplerate = sf.read(audiopath)
    #samples = samples[:, channel]
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1, sr=samplerate, alpha=alpha)
    sshow = sshow[2:, :]
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
    timebins, freqbins = np.shape(ims)
    
    ims = np.transpose(ims)
    # ims = ims[0:256, offset:offset+768] # 0-11khz, ~9s interval
    ims = ims[0:256, :] # 0-11khz, ~10s interval
    #print "ims.shape", ims.shape
    
    image = Image.fromarray(ims) 
    image = image.convert('L')
    image.save(name)


data_path = "data/LibriSpeech/"
def calculate_spectrograms(path):
    dirs = [os.path.join(path,d) for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
    
    def isflac(name):
        return "flac"==name.split(".")[-1]

    print("Started parsing folders...")
    for dir in dirs:
        speakers = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]
        for speaker in speakers:
            speaker_dir = os.path.join(dir,speaker)
            subdirs = [os.path.join(speaker_dir,d) for d in os.listdir(speaker_dir) if os.path.isdir(os.path.join(speaker_dir,d))]
            for subdir in subdirs:
                # now get flac files and process them
                files = [os.path.join(subdir,f) for f in os.listdir(subdir) if isflac(os.path.join(subdir,f))]
                for file in files:
                    print("Processing file: %s" % file)
                    #plotstft(file, name=".".join((file.split(".")[0],"png")))
                    img_name=".".join((file.split(".")[0],"png"))
                    graph_spectrogram(file, img_name)


    print("Finished spectrograms!")

def calculate_spectrograms_flat(path):
    def isflac(name):
        return "flac"==name.split(".")[-1]

    print("Started parsing folder '" + path + "'...")
    # now get flac files and process them
    files = [os.path.join(path,f) for f in os.listdir(path) if isflac(os.path.join(path,f))]
    for file in files:
        print("Processing file: %s" % file)
        img_name=".".join((file.split(".")[0],"png"))
        graph_spectrogram(file, img_name)

    print("Finished spectrograms for folder '" + path + "'!")

    
def get_labels(path):
    dirs = [os.path.join(path,d) for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
    
    def ispng(name):
        return "png"==name.split(".")[-1]

    labels=[]
    data=[]
    print("Started parsing folders...")
    for dir in dirs:
        speakers = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]
        for speaker in speakers:
            speaker_dir = os.path.join(dir,speaker)
            subdirs = [os.path.join(speaker_dir,d) for d in os.listdir(speaker_dir) if os.path.isdir(os.path.join(speaker_dir,d))]
            for subdir in subdirs:
                # now get flac files and process them
                files = [os.path.join(subdir,f) for f in os.listdir(subdir) if ispng(os.path.join(subdir,f))]
                for file in files:
                    print("Processing file: ", file)
                    labels.append(speaker)
                    data.append(file)
                    
    print("Finished!")    
    return data, labels

def convert_rgba2rgb(img_name):
    img = Image.open(img_name)
    x = np.array(img)
    r, g, b, a = np.rollaxis(x, axis=-1)
    r[a == 0] = 255
    g[a == 0] = 255
    b[a == 0] = 255 
    x = np.dstack([r, g, b])
    res = Image.fromarray(x, 'RGB')
    res.save(img_name)


def convert_allfiles_to_rgb(path):
    dirs = [os.path.join(path,d) for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
    
    def ispng(name):
        return "png"==name.split(".")[-1]

    labels=[]
    data=[]
    print("Started parsing folders...")
    for dir in dirs:
        speakers = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]
        for speaker in speakers:
            speaker_dir = os.path.join(dir,speaker)
            subdirs = [os.path.join(speaker_dir,d) for d in os.listdir(speaker_dir) if os.path.isdir(os.path.join(speaker_dir,d))]
            for subdir in subdirs:
                files = [os.path.join(subdir,f) for f in os.listdir(subdir) if ispng(os.path.join(subdir,f))]
                for file in files:
                    print("Processing file: ", file)
                    convert_rgba2rgb(file)
                    
    print("Finished!") 

#convert_allfiles_to_rgb(data_path)

'''files, labels = get_labels(data_path)
thefile = open('%sLibriData.txt' % data_path , 'w')
for i in range(len(files)):
    thefile.write("%s\t%s\n" % (files[i], labels[i]))
thefile.close()'''

#calculate_spectrograms(data_path)

