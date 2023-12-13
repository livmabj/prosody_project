# prosody_project

This project contains two models, one that process data as words and audio, and another that process data as words.
Only the LSTM-part of the model-scripts are intended to be used.

Necessary to run the models is also the dictionary_corpus in yedetere et al. (2023)

The models can be run using these commands:


python audio_main.py --data [DATA] --nlayers 2 --nhid 600 --emsize 300 --lr 5.0 --batch_size 20 --dropout 0.4 --seed 1001 --model LSTM --save audio_models/lstm.pt --log audio_models/lstm.log

python word_main.py --data [DATA] --nlayers 2 --nhid 600 --emsize 600 --lr 5.0 --batch_size 20 --dropout 0.4 --seed 1001 --model LSTM --save word_models/lstm.pt --log word_models/lstm.log

python audio_main.py --data [FINETUNEDATA] --nlayers 2 --nhid 600 --emsize 300 --lr 0.5 --batch_size 20 --dropout 0.5 --seed 1001 --model LSTM --load audio_models/lstm.pt --save audio_models/lstm-finetuned.pt --log audio_models/lstm-finetuned.log --finetune

python word_main.py --data [FINETUNEDATA] --nlayers 2 --nhid 600 --emsize 600 --lr 0.5 --batch_size 20 --dropout 0.5 --seed 1001 --model LSTM --load word_models/lstm.pt --save word_models/lstm-finetuned.pt --log word_models/lstm-finetuned.log --finetune

# Acknowledgements

This project is an extension of the model https://github.com/adityayedetore/lm-povstim-with-childes
yedetore et al. (2023)

All files that are adaptations of the files in that project begin with 'audio_' or 'word_' and has the licence on top.
The additions that have been made to the models are marked with # Addition

make_crepe.py and wav2vec_pipe.py are not adaptations.

[How poor is the stimulus? Evaluating hierarchical generalization in neural networks trained on child-directed speech](https://aclanthology.org/2023.acl-long.521) (Yedetore et al., ACL 2023)
