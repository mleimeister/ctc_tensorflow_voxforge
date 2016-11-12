# ctc_tensorflow_voxforge
Simple example how to use tensorflow's CTC loss with a BLSTM network and batch processing trained on a small number of Voxforge speech data.

The connectionist temporal classification (CTC) loss function was introduced in [[1](http://www.cs.toronto.edu/~graves/icml_2006.pdf)] for labelling unsegmented sequences. In contrast to other approaches for speech recognition, no a priori alignment of input speech features and target labels has to be known. The example shows how the CTC implementation from tensorflow can be used on top of a one-layer BLSTM network. Furthermore, batch processing using several time series with different length is applied.

Inspired by the examples of [igormq](https://github.com/igormq/ctc_tensorflow_example) and [kdavis-mozilla](https://github.com/kdavis-mozilla/DeepSpeech).

### download_voxforge_data.py

A number of example files is downloaded from the [Voxforge repository](http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/) that contain audio recordings and text transcriptions of short sentences. 

### generate_voxforge_txt_files.py
The prompts that are contained in one file in the Voxforge download are transformed into one txt file per example used later for generating the prediction targets. 

### generate_voxforge_training_data.py

As input feautures for speech recognition, Mel frequency cepstral coefficients (MFCCs) are extracted using [python_speech_features](http://python-speech-features.readthedocs.io/en/latest/). They are assembled into a batched format with the target character level annotations for subsequent training.

### train_ctc_voxforge.py

A single layer bidirectional LSTM network is trained to predict the transcriptions from the audio features. Every 10 epochs, an example batch is decoded and printed for comparison with the target. Over the course of 200 epochs, the training error drops to 10%. As expected, due to the very limited amount of data, the network does not generalize well to a hold-out batch, so that the validation error oscillates between 70%-80%. For this to work a probably much deeper network with much more data would be necessary, see e.g. [[2](https://arxiv.org/pdf/1412.5567v2.pdf)].


## References

[1] Alex Graves et al.: Connectionist Temporal Classification: Labelling Unsegmented
Sequence Data with Recurrent Neural Networks. ICML 2006. [pdf](http://www.cs.toronto.edu/~graves/icml_2006.pdf)

[2] Awni Hannun et al.: Deep Speech: Scaling up end-to-end speech recognition. [pdf](https://arxiv.org/pdf/1412.5567v2.pdf)
