import os
import numpy as np
import pickle
import scipy.io.wavfile as wav
from python_speech_features import fbank, mfcc

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

voxforge_data_dir = './Voxforge'

# Some configs
num_features = 13
# Accounting the 0th indice +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1


def list_files_for_speaker(speaker, folder):
    """
    Generates a list of wav files for a given speaker from the voxforge dataset.
    Args:
        speaker: substring contained in the speaker's folder name, e.g. 'Aaron'
        folder: base folder containing the downloaded voxforge data

    Returns: list of paths to the wavfiles
    """

    speaker_folders = [d for d in os.listdir(folder) if speaker in d]
    wav_files = []

    for d in speaker_folders:
        for f in os.listdir(os.path.join(folder, d, 'wav')):
            wav_files.append(os.path.abspath(os.path.join(folder, d, 'wav', f)))

    return wav_files


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), xrange(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape


def extract_features_and_targets(wav_file, txt_file):
    """
    Extract MFCC features from an audio file and target character annotations from
    a corresponding text transcription
    Args:
        wav_file: audio wav file
        txt_file: text file with transcription

    Returns:
        features, targets, sequence length, original text transcription
    """

    fs, audio = wav.read(wav_file)

    features = mfcc(audio, samplerate=fs, lowfreq=50)

    mean_scale = np.mean(features, axis=0)
    std_scale = np.std(features, axis=0)

    features = (features - mean_scale[np.newaxis, :]) / std_scale[np.newaxis, :]

    seq_len = features.shape[0]

    # Readings targets
    with open(txt_file, 'rb') as f:
        for line in f.readlines():
            if line[0] == ';':
                continue

            # Get only the words between [a-z] and replace period for none
            original = ' '.join(line.strip().lower().split(' ')).replace('.', '').replace("'", '').replace('-', '').replace(',','')
            targets = original.replace(' ', '  ')
            targets = targets.split(' ')

    # Adding blank label
    targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

    # Transform char into index
    targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX
                          for x in targets])

    # shape (None, num_steps, num_features)
    features = np.asarray(features[np.newaxis, :])

    return features, targets, seq_len, original


def make_batched_data(wav_files, batch_size=4):
    """
    Generate batches of data given a list of wav files from the downloaded Voxforge data.
    Args:
        wav_files: list of wav files
        batch_size: batch size

    Returns:
        batched data, original text transcriptions
    """

    batched_data = []
    original_targets = []
    num_batches = int(np.floor(len(wav_files) / batch_size))

    for n_batch in xrange(num_batches):

        batch_features = []
        batch_targets = []
        batch_seq_len = []
        batch_original = []

        for f in wav_files[n_batch * batch_size: (n_batch+1) * batch_size]:

            txt_file = f.replace('/wav/', '/txt/').replace('.wav', '.txt')
            features, targets, seq_len, original = extract_features_and_targets(f, txt_file)

            batch_features.append(features)
            batch_targets.append(targets)
            batch_seq_len.append(seq_len)
            batch_original.append(original)

        # Creating sparse representation to feed the placeholder
        batch_targets = sparse_tuple_from(batch_targets)
        max_length = max(batch_seq_len)

        padded_features = np.zeros(shape=(batch_size, max_length, batch_features[0].shape[2]), dtype=np.float)

        for i, feat in enumerate(batch_features):
            padded_features[i, :feat.shape[1], :] = feat

        batched_data.append((padded_features, batch_targets, batch_seq_len))
        original_targets.append(batch_original)

    return batched_data, original_targets


if __name__ == '__main__':

    wav_files = list_files_for_speaker('Aaron', voxforge_data_dir)

    batched_data, original_targets = make_batched_data(wav_files, batch_size=4)

    with open('train_data_batched.pkl', 'wb') as f:
        pickle.dump(batched_data, f)

    with open('original_targets_batched.pkl', 'wb') as f:
        pickle.dump(original_targets, f)

