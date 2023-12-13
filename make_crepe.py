from tqdm import tqdm
import crepe
from scipy.io import wavfile
import pickle
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
import sys

# file with .wav files
wav_path = sys.argv[1]

# The output from wav2vec 2.0, organized as a dictionary
wtv_output = sys.argv[2]

# file to output list of frequencies to
outputfile = sys.argv[3]

wavfiles = [f for f in listdir(wav_path) if isfile(join(wav_path, f))]

out = []

with open(wtv_output, "rb") as fin:
    dictfile = pickle.load(fin)
stop = 0
for id, dict in tqdm(dictfile.items(), desc="Processing Files", unit="file"):
    if id+".wav" in wavfiles: # matching the wavfile to id in the dictionary
        word_offsets = dict["word_offsets"]
        sent = dict["sentence"].split()
        # reading the wavfile
        sr, audio = wavfile.read(wav_path+id+".wav")
        time, frequency, confidence, activation = crepe.predict(audio,sr,viterbi=True, step_size=100, model_capacity="medium") # stepsize 100 ms
        # the mean frequency of the sentence
        full_average = np.mean(frequency)
        norm_freqs = []
        # going through all timesteps
        for start in time:
            # going through all words in the sentence
            for word in word_offsets:
                # match timestep to start time
                if start == round(word["start_time"],1):
                    end = round(word["end_time"],1)
                    # getting a list of the frequencies between start and end of word
                    herz = frequency[int(start*10):int(end*10)]
                    if herz.size != 0:
                        # the mean of the list is the word frequency
                        mean_of_word = np.mean(herz)
                        norm_freqs.append(mean_of_word-full_average) # normalization
                    else:
                        mean_of_word = 0 # some faulty recordings are 0-ed

                        norm_freqs.append(0)

        finprod = torch.Tensor(np.array(norm_freqs))
        out.append(finprod)



with open(outputfile, "wb") as fut:
    pickle.dump(out,fut)