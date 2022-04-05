
# %%
import argparse
import json
import audiomentations as am
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle as pkl
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from preprocess import augment_signal, segment_signal, represent_signal, represent_dataset, segment_dataset

from praudio import utils

from dataset import split_dataset, annotate_dataset
from loaders import load_mat_representation
from plot import plot_confusion_matrix, plot_history
from train import build_perceptron, train_model

parser = argparse.ArgumentParser(description='Arguments algo')

parser.add_argument('--ip', type=str, action='store', dest='ip',
                    required=False, help='IP', default=None)

parser.add_argument('-c', type=int, action='store', dest='coeff', required=False, help='Coeficientes',
                    default=None)

parser.add_argument('-a', type=int, action='store', dest='augmentation', required=False, help='Augmentation',
                    default=None)

parser.add_argument('-s', type=int, action='store', dest='segment', required=False, help='Segment ime',
                    default=None)

parser.add_argument('-o', type=float, action='store', dest='overlap', required=False, help='Overalp data',
                    default=None)

parser.add_argument('-b', type=str, action='store', dest='base', required=False, help='Dataset',
                    default=None)


args, _ = parser.parse_known_args()

# %%
BASE_DATASETS = '/src/datasets'
ANNOTATE_DIR = '/src/tcc/dataset'
MODELS_DIR = '/src/tcc/models'
DATASET_DIR = args.base or 'base_portuguese'

ANNOTATE_DATASET = False
SPLIT_DATESET = False
SEGMENT_TEST = False
SEGMENT_TRAIN = False
SEGMENT_VALID = False
REPRESENT_TEST = False
REPRESENT_TRAIN = False
REPRESENT_VALID = False

# MODEL ARCHITECTURE
MODEL_DENSE_1 = 90
MODEL_DROPOUT_1 = 0
MODEL_DENSE_2 = 0
MODEL_DROPOUT_2 = 0
MODEL_DENSE_3 = 0

# MODEL TRAINING
EPOCHS = 40
BATCH_SIZE = 32
PATIENCE = 5
LEARNING_RATE = 0.0001

# SEGMENTATION && REPRESENTATION
SAMPLE_RATE = 24000
SEGMENT_LENGTH = args.segment or 26
OVERLAP_SIZE = args.overlap or 0.0
AUGMENT_SIZE = args.augmentation or 30
MFCC_COEFF = args.coeff or 40
MFCC_N_FFT = 2048
MFCC_HOP_LENGTH = 512

# %%
if ANNOTATE_DATASET:
    annotate_dataset(f'{BASE_DATASETS}/{DATASET_DIR}',
                     f'{ANNOTATE_DIR}/{DATASET_DIR}',
                     SAMPLE_RATE)

# %%
if SPLIT_DATESET:
    split_dataset(f'{ANNOTATE_DIR}/{DATASET_DIR}',
                  f'{ANNOTATE_DIR}/{DATASET_DIR}',
                  validation=True)

# %%
BASE_TRANSFORM = [
    am.Trim(top_db=20, p=1),
    am.Normalize(p=1),
]

TRAIN_TRANSFORM = [
    am.AddGaussianSNR(min_snr_in_db=24, max_snr_in_db=40, p=0.8),
    am.HighPassFilter(min_cutoff_freq=60, max_cutoff_freq=100, p=0.8),
    am.LowPassFilter(min_cutoff_freq=3400, max_cutoff_freq=4000, p=0.8),
    am.TimeStretch(min_rate=0.75, max_rate=2,
                   leave_length_unchanged=False, p=0.5),
]

# %%
BASE_DIR = f'{ANNOTATE_DIR}/{DATASET_DIR}/SEG_{SEGMENT_LENGTH}_OVERLAP_{int(OVERLAP_SIZE*100)}'

if SEGMENT_TEST:
    segment_dataset(f'{ANNOTATE_DIR}/{DATASET_DIR}/test',
                    f'{BASE_DIR}/test',
                    base_trans=BASE_TRANSFORM,
                    overlap_size=OVERLAP_SIZE,
                    segment_length=SEGMENT_LENGTH)

# %%
if SEGMENT_VALID:
    segment_dataset(f'{ANNOTATE_DIR}/{DATASET_DIR}/valid',
                    f'{BASE_DIR}/valid',
                    base_trans=BASE_TRANSFORM,
                    overlap_size=OVERLAP_SIZE,
                    segment_length=SEGMENT_LENGTH)

# %%
if SEGMENT_TRAIN:
    segment_dataset(f'{ANNOTATE_DIR}/{DATASET_DIR}/train',
                    f'{BASE_DIR}/train',
                    base_trans=BASE_TRANSFORM,
                    extra_trans=TRAIN_TRANSFORM,
                    overlap_size=OVERLAP_SIZE,
                    augment_size=AUGMENT_SIZE,
                    segment_length=SEGMENT_LENGTH)

# %% REPRESENTATION
if REPRESENT_TEST:
    mat_dict_test = represent_dataset(f'{BASE_DIR}/test',
                                      f'{BASE_DIR}/MFCC_{MFCC_COEFF}/test',
                                      n_mfcc=MFCC_COEFF,
                                      n_fft=MFCC_N_FFT,
                                      hop_length=MFCC_HOP_LENGTH)
# %%
if REPRESENT_VALID:
    mat_dict_valid = represent_dataset(f'{BASE_DIR}/valid',
                                       f'{BASE_DIR}/MFCC_{MFCC_COEFF}/valid',
                                       n_mfcc=MFCC_COEFF,
                                       n_fft=MFCC_N_FFT,
                                       hop_length=MFCC_HOP_LENGTH)
# %%
if REPRESENT_TRAIN:
    mat_dict_train = represent_dataset(f'{BASE_DIR}/train',
                                       f'{BASE_DIR}/MFCC_{MFCC_COEFF}/train',
                                       n_mfcc=MFCC_COEFF,
                                       n_fft=MFCC_N_FFT,
                                       hop_length=MFCC_HOP_LENGTH)

# %%

# %% LOAD REPRESENTATION
if not REPRESENT_TEST:
    mat_dict_test = load_mat_representation(
        f'{BASE_DIR}/MFCC_{MFCC_COEFF}/test/representation.mat')


if not REPRESENT_TRAIN:
    mat_dict_train = load_mat_representation(
        f'{BASE_DIR}/MFCC_{MFCC_COEFF}/train/representation.mat')

if not REPRESENT_VALID:
    mat_dict_valid = load_mat_representation(
        f'{BASE_DIR}/MFCC_{MFCC_COEFF}/valid/representation.mat')

# %% NP.ARRAY
unique_labels = list(set(mat_dict_train['label']))

X_train = np.array(mat_dict_train['representation'])
y_train = np.array(mat_dict_train['label'])
X_valid = np.array(mat_dict_valid['representation'])
y_valid = np.array(mat_dict_valid['label'])
X_test = np.array(mat_dict_test['representation'])
y_test = np.array(mat_dict_test['label'])
# %% SCALER
se = StandardScaler()

X_train_rep = se.fit_transform(
    X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_valid_rep = se.transform(
    X_valid.reshape(-1, X_valid.shape[-1])).reshape(X_valid.shape)
X_test_rep = se.transform(
    X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)


# %% BUILD MODEL
model = build_perceptron(output_size=len(unique_labels),
                         shape_size=X_train_rep.shape,
                         dense1=MODEL_DENSE_1,
                         dropout1=MODEL_DROPOUT_1,
                         dense2=MODEL_DENSE_2,
                         dropout2=MODEL_DROPOUT_2,
                         dense3=MODEL_DENSE_3,
                         learning_rate=LEARNING_RATE)

# %% MODEL SUMMARY
model_arch = model.to_json()
model.summary()

# %% TRAIN MODEL
history = train_model(model,
                      epochs=EPOCHS,
                      batch_size=BATCH_SIZE,
                      patience=PATIENCE,
                      X_train=X_train_rep,
                      y_train=y_train,
                      X_validation=X_valid_rep,
                      y_validation=y_valid)


# %%
test_loss, test_acc = model.evaluate(X_test_rep,
                                     y_test,
                                     verbose=2)

# %%
y_pred = model.predict(X_test_rep)
y_pred = np.argmax(y_pred, axis=1)

confusion = tf.math.confusion_matrix(y_pred, y_test)

# %% SAVING PROCESS
timestamp = int(time.time())

save_foldername = f'{MODELS_DIR}/{DATASET_DIR}/SEG_{SEGMENT_LENGTH}/MFCC_{MFCC_COEFF}/{timestamp}_{test_acc * 100}'

utils.create_dir_hierarchy(save_foldername)

with open(f'{save_foldername}/model_architecture.json', 'w') as f:
    f.write(model_arch)

model.save(
    f'{save_foldername}/model.h5')

pkl.dump(se, open(
    f'{save_foldername}/scaler.pkl', 'wb'))

plot_history(history,
             save_path=f'{save_foldername}')

plot_confusion_matrix(confusion.numpy(),
                      size=len(unique_labels),
                      save_path=f'{save_foldername}')

# %%
overview = {
    'dataset_dir': DATASET_DIR,
    'classes': len(unique_labels),

    'segment_length': SEGMENT_LENGTH,
    'sample_rate': SAMPLE_RATE,
    'augment_size': AUGMENT_SIZE,
    'overlap_size': OVERLAP_SIZE,

    'train_shape': X_train_rep.shape,
    'valid_shape': X_valid_rep.shape,
    'test_shape': X_test_rep.shape,

    'scores': {
        'test_loss': test_loss,
        'test_acc': test_acc,

        'train_loss': history.history['loss'][-1],
        'train_acc': history.history['accuracy'][-1],

        'valid_loss': history.history['val_loss'][-1],
        'valid_acc': history.history['val_accuracy'][-1],
    },

    'training_params': {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'patience': PATIENCE,
        'learning_rate': LEARNING_RATE,
    },


    'representation': {
        'name': 'MFCC',
        'n_mfcc': MFCC_COEFF,
        'n_fft': MFCC_N_FFT,
        'hop_length': MFCC_HOP_LENGTH,
    },

    'classes': {
        mat_dict_test['mapping'][index]: mat_dict_test['label'][index] for index, _ in enumerate(mat_dict_test['mapping'])}
}


with open(f'{save_foldername}/overview.json', 'w') as f:
    f.write(json.dumps(overview))
