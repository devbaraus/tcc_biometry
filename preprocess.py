# %%
import os
import shutil
import numpy as np
import audiomentations as am
import matplotlib.pyplot as plt
import librosa
from praudio import utils

from joblib import Parallel, delayed

import os
import audiomentations as am
import pandas as pd
import librosa
from praudio import utils
import scipy.io as sio

from utils import merge_dicts


def augment_signal(signal: np.array, sample_rate: int, transformations: list, augment_size: int):
    """
    """
    augment_composition = am.Compose(transformations)

    augmented_signals = []

    for i in range(augment_size):
        augmented_signal = augment_composition(signal,
                                               sample_rate)

        augmented_signals.append(augmented_signal)

    return augmented_signals


def represent_signal(signal: np.array, sample_rate: int, plot: bool = False, **mfcc_params):
    """
    """

    mfcc = []

    def _plot(x_label='Frame Index', y_label='Index', cmap='magma', size=(10, 6)):
        if size:
            plt.figure(figsize=(10, 6), frameon=True)

        plt.imshow(mfcc,
                   origin='lower',
                   aspect='auto',
                   cmap=cmap,
                   interpolation='nearest')

        plt.title('MFCC')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()
        plt.show()
        plt.savefig(f'/src/tcc/plot/mfcc.png', dpi=300)
        plt.close()

    mfcc = librosa.feature.mfcc(signal,
                                sr=sample_rate,
                                **mfcc_params)

    if plot:
        _plot()

    return mfcc


def segment_signal(signal: np.array, sample_rate: int, segment_length: int, overlap_size: float = 0, plot=False):
    """
    """
    segments = []
    seg_positon = []

    def _plot_segments():
        shutil.rmtree('/src/tcc/plot', ignore_errors=True)
        os.mkdir('/src/tcc/plot')

        duration = len(signal) / sample_rate
        time = np.arange(0, duration, 1/sample_rate)

        for i in range(len(segments)):
            fake_signal = np.zeros(len(signal))
            fake_signal[seg_positon[i][0]:seg_positon[i][1]] = segments[i]

            plt.plot(time, signal)
            plt.plot(time, fake_signal)

            plt.margins(x=0)
            plt.ylim(1, -1)
            plt.show()
            plt.savefig(f'/src/tcc/plot/segment_{i}.png', dpi=300)
            plt.close()

    overlap = 1 - overlap_size
    size_segment = sample_rate * segment_length
    size_overlap_segment = size_segment * overlap
    qtd_segments = 0
    flag = 1
    start_segment = 0

    while flag == 1:
        if start_segment + size_segment > len(signal):
            flag = 0
        else:
            qtd_segments = qtd_segments + 1
            start_segment = start_segment + size_overlap_segment

    for i in list(range(0, qtd_segments)):
        start_seg = int(i * sample_rate * segment_length * overlap)
        end_seg = int(start_seg + sample_rate * segment_length)

        segment_audio = signal[start_seg:end_seg]

        seg_positon.append([start_seg, end_seg])
        segments.append(segment_audio)

    if plot:
        _plot_segments()

    return segments


def segment_dataset(input_dir: str,
                    output_dir: str,
                    base_trans: list,
                    extra_trans: list = [],
                    augment_size: int = 0,
                    overlap_size: float = 0.0,
                    segment_length: int = 1):
    df = pd.read_csv(f'{input_dir}/metadata.csv')

    base_dict = {
        "mapping": [],
        "label": [],
        "sample_rate": [],
        "length": [],
        "filename": [],
        "aug_filename": [],
        "transformations": [],
    }

    def _run(i: int):
        df_dict = base_dict.copy()

        row = df.iloc[i, :]

        src_filename = utils.remove_extension_from_file(row['filename'])

        signal, sample_rate = librosa.load(
            f'{input_dir}/audio/{row["filename"]}', sr=row["sample_rate"], mono=True)

        ### save original segments ###
        transformations_name = '-'.join([
            trans.__class__.__name__ for trans in base_trans])

        augmented_signal = augment_signal(signal,
                                          sample_rate,
                                          base_trans,
                                          1)[0]

        segments = segment_signal(augmented_signal,
                                  sample_rate,
                                  segment_length=segment_length,
                                  overlap_size=overlap_size,
                                  plot=False)

        for indexI, segment in enumerate(segments):
            df_dict['mapping'].append(row['mapping'])
            df_dict['label'].append(row['label'])
            df_dict['sample_rate'].append(row['sample_rate'])
            df_dict['length'].append(len(segment)/sample_rate)

            seg_filename = f'{src_filename}_{indexI}.wav'

            df_dict['filename'].append(row['filename'])
            df_dict['aug_filename'].append(seg_filename)

            sio.wavfile.write(f'{output_dir}/audio/{seg_filename}',
                              sample_rate,
                              segment)

            df_dict['transformations'].append(transformations_name)

        ### save augmented segments ###
        transformations = [*base_trans, *extra_trans]

        transformations_name = '-'.join([
            trans.__class__.__name__ for trans in transformations])

        augmented_signals = augment_signal(signal,
                                           sample_rate,
                                           transformations,
                                           augment_size)

        for indexI, augmented_signal in enumerate(augmented_signals):
            segments = segment_signal(augmented_signal,
                                      sample_rate,
                                      segment_length=segment_length,
                                      overlap_size=0,
                                      plot=False)

            for indexJ, segment in enumerate(segments):
                df_dict['mapping'].append(row['mapping'])
                df_dict['label'].append(row['label'])
                df_dict['sample_rate'].append(row['sample_rate'])
                df_dict['length'].append(len(segment)/sample_rate)

                aug_filename = f'{src_filename}_{indexI}_{indexJ}.wav'

                df_dict['filename'].append(row['filename'])
                df_dict['aug_filename'].append(aug_filename)

                sio.wavfile.write(f'{output_dir}/audio/{aug_filename}',
                                  sample_rate,
                                  segment)

                df_dict['transformations'].append(transformations_name)

        return df_dict

    utils.create_dir_hierarchy(f'{output_dir}/audio')

    dicts = Parallel(n_jobs=-1)(delayed(_run)(i) for i in range(len(df)))

    df_dict = merge_dicts(base_dict, *dicts)

    pd.DataFrame.from_dict(df_dict).to_csv(f'{output_dir}/metadata.csv',
                                           index=False)


def represent_dataset(input_dir, output_dir, **mfcc_params):
    df = pd.read_csv(f'{input_dir}/metadata.csv')

    mat_dict = df.to_dict(orient='list')

    mat_dict['representation'] = []

    # for i in [2]:
    # for i in range(len(df)):
    def _run(i: int):
        row = df.iloc[i, :]

        signal, sample_rate = librosa.load(
            f'{input_dir}/audio/{row["aug_filename"]}',
            sr=row["sample_rate"],
            mono=True)

        representation = represent_signal(signal,
                                          sample_rate,
                                          plot=False,
                                          **mfcc_params)

        return representation

    utils.create_dir_hierarchy(f'{output_dir}')

    representations = Parallel(n_jobs=-1)(delayed(_run)(i)
                                          for i in range(len(df)))

    mat_dict['representation'] = representations
    sio.savemat(f'{output_dir}/representation.mat', mat_dict)

    return mat_dict
