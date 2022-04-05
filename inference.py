# %%
import os.path
import argparse
import json
import os
import shutil
import time
import numpy as np
import pandas as pd
import tensorflow.keras as keras
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import f1_score
import librosa
from praudio import utils

import audiomentations as am

from loaders import load_mat_representation
from plot import plot_confusion_matrix
from preprocess import augment_signal, pipeline_signal, represent_signal, segment_dataset, segment_signal
from dataset import annotate_inference

# %%
parser = argparse.ArgumentParser(description='Arguments algo')

parser.add_argument('-m', type=str,
                    action='store',
                    dest='model',
                    required=False,
                    help='Model Path',
                    default='/src/tcc/models/base_portuguese_40/SEG_2/MFCC_40/1649037627_86.50793433189392')

parser.add_argument('-i',
                    type=str,
                    action='store',
                    dest='input',
                    required=False,
                    help='Input Path',
                    default='/src/tcc/dataset/inference')

parser.add_argument('-c',
                    type=int,
                    action='store',
                    dest='cls',
                    required=False,
                    help='Class',
                    default='30')


args, _ = parser.parse_known_args()

BASE_TRANSFORM = [
    am.Trim(top_db=20, p=1),
    am.Normalize(p=1),
]
# %%
# annotate_inference('/src/datasets/ifgaudio',
#                    '/src/tcc/dataset/inference',
#                    '/src/tcc/models/base_portuguese_40/SEG_1/MFCC_18/1649037587_71.875',
#                    '/src/tcc/catalog.csv')
# %%
overview = open(f'{args.model}/overview.json', 'r').read()
overview = json.loads(overview)

# %%
extension = os.path.splitext(args.input)[1]

# %%
if extension == '.mat':
    mat_dict_test = load_mat_representation(args.input)

    X_inf = np.array(mat_dict_test['representation'])
    y_inf = np.array(mat_dict_test['label'])

elif extension == '.wav':
    X_inf = pipeline_signal(args.input,
                            overview['sample_rate'],
                            overview["segment_length"],
                            BASE_TRANSFORM,
                            n_mfcc=overview['representation']['n_mfcc'],
                            n_fft=overview['representation']['n_fft'],
                            hop_length=overview['representation']['hop_length'])

    y_inf = [args.cls] * len(X_inf)
elif not os.path.isfile(args.input):
    X_inf = None
    y_inf = None

    if not os.path.isdir(f'{args.input}/audio') and not os.path.isfile(f'{args.input}/metadata.csv'):
        raise Exception(
            f'Input path must contain an audio folder and metadata.csv file')

    df = pd.read_csv(f'{args.input}/metadata.csv')

    for index, row in df.iterrows():
        X_inf_aux = pipeline_signal(f'{args.input}/audio/{row["filename"]}',
                                    overview['sample_rate'],
                                    overview["segment_length"],
                                    BASE_TRANSFORM,
                                    n_mfcc=overview['representation']['n_mfcc'],
                                    n_fft=overview['representation']['n_fft'],
                                    hop_length=overview['representation']['hop_length'])

        y_inf_aux = [row['label']] * len(X_inf_aux)

        if X_inf is None:
            X_inf = X_inf_aux
            y_inf = y_inf_aux
        elif X_inf.shape[1:] == X_inf_aux.shape[1:]:
            X_inf = np.concatenate((X_inf, X_inf_aux))
            y_inf = np.concatenate((y_inf, y_inf_aux))

# %%
model_json = open(f'{args.model}/model_architecture.json', 'r').read()
model = keras.models.model_from_json(model_json)
model.load_weights(f'{args.model}/model.h5')

model.summary()
# %%
se_load = open(f'{args.model}/scaler.pkl', 'rb')
se = pkl.load(se_load)
se_load.close()

# %%
X_inf_rep = se.transform(
    X_inf.reshape(-1, X_inf.shape[-1])).reshape(X_inf.shape)

# %%
y_pred = model.predict(X_inf_rep)
y_pred_argmax = np.argmax(y_pred, axis=1)
# %%
f1_micro = f1_score(y_inf, y_pred_argmax, average='micro')
f1_macro = f1_score(y_inf, y_pred_argmax, average='macro')

# %%
table_of_truth = np.dstack((y_inf, y_pred_argmax, np.max(y_pred, axis=1)))
# %%
folder = args.model.replace('/models/', '/inference/')
utils.create_dir_hierarchy(folder)

# %%
unique_labels = overview['classes'].keys()

confusion = tf.math.confusion_matrix(y_pred_argmax, y_inf)
plot_confusion_matrix(confusion.numpy(),
                      size=len(unique_labels),
                      save_path=f'{folder}')
# %%
overview = {
    **overview,
    'inference': {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'table_of_truth': table_of_truth[0].tolist(),
    }
}

shutil.copy(f'{args.model}/model_architecture.json',
            f'{folder}/model_architecture.json')

with open(f'{folder}/overview.json', 'w') as f:
    f.write(json.dumps(overview))

# %%
