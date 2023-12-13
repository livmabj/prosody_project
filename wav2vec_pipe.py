# Let's see how to retrieve time steps for a model
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
import torch
from pathlib import Path
import pickle
import sys
from pydub import AudioSegment
from tqdm import tqdm
import scipy.io.wavfile



# import model, feature extractor, tokenizer
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")


train_hundred = 28539 # number of utterances
train_threesixty = 104014

input_folder = sys.argv[1]
outfile = sys.argv[2]

all_files = {}

wav_files = [file for file in Path(input_folder).iterdir() if file.is_file()]
print("Number of .wav files:", len(list(wav_files)))
for input_file_path in tqdm(wav_files, total=250):

    # keep the id to match with the frequency output at a later stage
    audio_id = input_file_path

    # load audio
    sample_rate, audio_input = scipy.io.wavfile.read(input_file_path)
    audio_input = audio_input.astype('d')

    # forward sample through model to get greedily predicted transcription ids
    input_values = feature_extractor(audio_input, return_tensors="pt", sampling_rate=16_000).input_values
    logits = model(input_values).logits[0]
    pred_ids = torch.argmax(logits, axis=-1)

    # retrieve word stamps
    outputs = tokenizer.decode(pred_ids, output_word_offsets=True)
    # compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
    time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate

    word_offsets = [
        {
            "word": d["word"],
            "start_time": round(d["start_offset"] * time_offset, 2),
            "end_time": round(d["end_offset"] * time_offset, 2),
        }
        for d in outputs.word_offsets
    ]

    all_files[audio_id] = {
                            "word_offsets": word_offsets
                            }

with open(outfile, 'wb') as f:  # open a pickle file
    pickle.dump(all_files, f)