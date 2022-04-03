import os
import shutil

import librosa
import pandas as pd
import scipy.io as sio
from praudio import utils
from sklearn.model_selection import train_test_split

DATASET_PATH = '../datasets/base_portuguese'
OUTPUTDIR_PATH = './dataset/base'
SAMPLES_TO_CONSIDER = 22050  # 1 sec. of audio
SEED = 42


def annotate_dataset(dataset_path: str, output_path: str):
    """
    It takes in a dataset path and outputs a metadata.csv file.

    :param dataset_path: The path to the dataset folder
    :type dataset_path: str
    :param output_path: The path where the audio files will be saved
    :type output_path: str
    """

    data = {
        "mapping": [],
        "label": [],
        "sample_rate": [],
        "length": [],
        "filename": [],
    }

    utils.create_dir_hierarchy(output_path + '/audio')

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're at sub-folder level
        if dirpath is not dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1]
            print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)

                data["mapping"].append(label)
                data['sample_rate'].append(sample_rate)
                data['length'].append(len(signal) / sample_rate)
                data["label"].append(i - 1)
                data["filename"].append(f)

                sio.wavfile.write(
                    f'{output_path}/audio/{f}', sample_rate, signal)
                print("{}: {}".format(file_path, i - 1))

    pd.DataFrame.from_dict(data).to_csv(
        f'{output_path}/metadata.csv', index=False)


def split_dataset(input_dataset: str, output_path: str, validation: bool = True):
    """
    It splits the dataset into train, test and validation subsets, and copies the audio files to the
    corresponding folders

    :param input_dataset: The path to the dataset folder
    :type input_dataset: str
    :param output_path: the path where the dataset will be created
    :type output_path: str
    :param validation: If True, the validation set will be created, defaults to True
    :type validation: bool (optional)
    """

    df = pd.read_csv(f'{input_dataset}/metadata.csv')

    labels = df['label'].tolist()

    subsets = {}

    X_train, X_test, X_valid, y_train, y_test, y_valid = [], [], [], [], [], []

    X_train, X_test, y_train, y_test = train_test_split(df,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=SEED,
                                                        stratify=labels)

    subsets['train'] = X_train
    subsets['test'] = X_test

    if validation:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                              y_train,
                                                              test_size=0.2,
                                                              random_state=SEED,
                                                              stratify=y_train)
        subsets['train'] = X_train
        subsets['valid'] = X_valid

    # check if all subsets have all labels
    if validation and not (list(set(y_valid)) == list(set(y_test)) == list(set(y_train))):
        raise BaseException('All subsets need to contain the same label')
    elif not (list(set(y_test)) == list(set(y_train))):
        raise BaseException('All subsets need to contain the same label')

    for key, value in subsets.items():
        shutil.rmtree(f'{output_path}/{key}', ignore_errors=True)

        utils.create_dir_hierarchy(f'{output_path}/{key}/audio')

        df_subset = pd.DataFrame.from_dict(value)
        df_subset.to_csv(f'{output_path}/{key}/metadata.csv', index=False)

        for audio in df_subset['filename'].tolist():
            src = f'{input_dataset}/audio/{audio}'
            dst = f'{output_path}/{key}/audio/{audio}'

            # if not os.path.exists(src):
            shutil.copy(src, dst)


# if __name__ == '__main__':
    # prepare_raw_dataset(DATASET_PATH, OUTPUTDIR_PATH)
    # split_dataset('/src/tcc_devbaraus/dataset/base', '/src/tcc_devbaraus/dataset', True)
