import pyaudio
import wave
import soundfile as sf
import audioop
from collections import deque
import os
import sys
import time
import math
from create_spectrograms import calculate_spectrograms_flat as get_spectros


# Microphone stream config.
CHUNK = 1024  # CHUNKS of bytes to read each time from mic
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
THRESHOLD = 2500  # The threshold intensity that defines silence
                  # and noise signal (an int. lower than THRESHOLD is silence).

SILENCE_LIMIT = 1  # Silence limit in seconds. The max ammount of seconds where
                   # only silence is recorded. When this time passes the
                   # recording finish es and the file is delivered.

PREV_AUDIO = 0.5  # Previous audio (in seconds) to prepend. When noise
                  # is detected, how much of previously recorded audio is
                  # prepended. This helps to prevent chopping the beggining
                  # of the phrase.

def listen_for_speech(path, threshold=THRESHOLD, num_phrases=-1):
    """
    Listens to Microphone, extracts phrases from it and sends it to 
    Google's TTS service and returns response. a "phrase" is sound 
    surrounded by silence (according to threshold). num_phrases controls
    how many phrases to process before finishing the listening process 
    (-1 for infinite). 
    """

    #Open stream
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print "* Listening mic. "
    audio2send = []
    cur_data = ''  # current chunk  of audio data
    rel = RATE/CHUNK
    slid_win = deque(maxlen=SILENCE_LIMIT * rel)
    #Prepend audio from 0.5 seconds before noise was detected
    prev_audio = deque(maxlen=PREV_AUDIO * rel) 
    started = False
    n = num_phrases
    response = []

    while (num_phrases == -1 or n > 0):
        cur_data = stream.read(CHUNK)
        slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))
        #print slid_win[-1]
        if(sum([x > THRESHOLD for x in slid_win]) > 0):
            if(not started):
                print "Starting record of phrase"
                started = True
            audio2send.append(cur_data)
        elif (started is True):
            print "Finished"
            # The limit was reached, finish capture and deliver.
            filename = save_speech(path, list(prev_audio) + audio2send, p)
            print("Saved to " + filename)


            # Reset all
            started = False
            slid_win = deque(maxlen=SILENCE_LIMIT * rel)
            prev_audio = deque(maxlen=0.5 * rel) 
            audio2send = []
            n -= 1
            print "Listening ..."
        else:
            prev_audio.append(cur_data)

    print "* Done recording"
    stream.close()
    p.terminate()

    return response


def save_speech(path, data, p):
    """ Saves mic data to FLAC file. Returns filename of saved 
        file """

    filename = path + str(int(time.time()))
    wavname = save_speech_wav(filename, data, p)

    wavdata, samplerate = sf.read(wavname)
    sf.write(filename + '.flac', wavdata, samplerate)
    os.remove(wavname)

    return filename + '.flac'


def save_speech_wav(fname, data, p):
    """ Saves mic data to temporary WAV file. Returns filename of saved 
        file """

    # writes data to WAV file
    data = ''.join(data)
    wf = wave.open(fname + '.wav', 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)  # TODO make this value a function parameter?
    wf.writeframes(data)
    wf.close()
    return fname + '.wav'


def record_audio(path):
	listen_for_speech(path, num_phrases=5)  # listen to mic.
	get_spectros(path)


if(__name__ == '__main__'):
    
    path = 'recordings_train/'
    if len(sys.argv)>1:
    	arg = sys.argv[1]
    	if not os.path.isdir(arg):
    		print('Directory "' + arg + '" does not exist.\nUsing default path "recordings_train/" instead.')
    	else:
    		path=arg


    record_audio(path)