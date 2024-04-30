import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

import librosa.display
import librosa

import os
import struct
from scipy.io import wavfile as wav
import IPython.display as ipd
from tqdm import tqdm
import warnings

import glob
import soundfile as sf
from tqdm import tqdm

from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler, OneHotEncoder
import tensorflow as tf
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization

from datetime import datetime


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

warnings.filterwarnings("ignore")

def load_data(file_name):
    """Returns a pandas dataframe from a csv file."""
    return pd.read_csv(file_name)

path_ = '../input/urbansound8k/'
path_csv = path_+'UrbanSound8K.csv'
metadata = load_data(path_csv)

metadata.head(10)

metadata.tail()

filename = "../input/urbansound8k/fold1/101415-3-0-2.wav"
audiodata,sr=librosa.load(filename)


# Showing our first wave of the audio data:

librosa.display.waveshow(audiodata,sr=sr)
ipd.Audio(filename) # Audio clip of the first audio data


metadata.info()

metadata.describe().T

metadata.isnull().sum()

# Check whether the dataset is imbalanced:

metadata['class'].value_counts()

ax = sns.histplot(y='class',data = metadata, hue="class", multiple="stack")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.xticks(rotation=45);
sns.despine()
plt.show()

class_dict = metadata['class'].value_counts(normalize=True)
classes = list(np.unique(metadata['class']))
fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08, fontsize=18)
ax.pie(class_dict, labels=classes, autopct='%1.1f%%', shadow=False, startangle=180)
ax.axis('equal')
plt.savefig("distribution_class")
plt.show(block=False)

# Let's add metadata to the path of audio files in the ‘Audio’ folder:

metadata['path_file'] = path_+'fold'+metadata.fold.astype(str)+'/'+metadata['slice_file_name']
metadata.head()

file0 = metadata.path_file[0]


df = metadata[['path_file','class']]

df.head()

# Time Series on classes:
def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

def plot_waves(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(35,15))
    
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(2,5,i)
        librosa.display.waveshow(np.array(f),sr=22050, color='orange', alpha=0.7,where = 'post')
        plt.title(n.title())
        i += 1
        
    fig.suptitle('Figure 1: Waveplot, Time Series', fontsize=18)
    plt.savefig("time_series")
    plt.tight_layout()
    plt.show()
  def plot_spec(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(35,15))
    
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(2,5,i)
        d =librosa.amplitude_to_db(np.abs(librosa.stft(f)), ref=np.max)
        librosa.display.specshow(d, y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title(n.title())
        i += 1
        
    fig.suptitle('Figure 2: Linear-Frequency Power Spectrogram', fontsize=18)
    plt.savefig("spectrogram")
    plt.tight_layout()
    plt.show()
    
    sound_file_paths = ["../input/urbansound8k/fold5/100852-0-0-0.wav","../input/urbansound8k/fold10/100648-1-0-0.wav",
                   "../input/urbansound8k/fold5/100263-2-0-117.wav","../input/urbansound8k/fold5/100032-3-0-0.wav",
                   "../input/urbansound8k/fold3/103199-4-0-3.wav","../input/urbansound8k/fold10/102857-5-0-0.wav",
                   "../input/urbansound8k/fold1/102305-6-0-0.wav","../input/urbansound8k/fold1/103074-7-1-0.wav",
                   "../input/urbansound8k/fold7/102853-8-0-2.wav","../input/urbansound8k/fold7/101848-9-0-9.wav"]
classes = list(np.unique(df['class']))

raw_sounds = load_sound_files(sound_file_paths)
plot_waves(classes,raw_sounds)
plot_spec(classes,raw_sounds)

# Feature set:
# This file is of a dog bark:

y,sr = librosa.load(file0)

# stft -> Short-time Fourier transform:(The STFT represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short overlapping windows)
S = librosa.stft(y) 
# Mel-frequency cepstral coefficients:
mfccs = librosa.feature.mfcc(y, sr, n_mfcc=40) 
# Compute a mel-scaled spectrogram:
melspectrogram =librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,fmax=8000) 
# Compute a chromagram from a waveform or power spectrogram:
chroma_stft=librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=40)
# Constant-Q chromagram:
chroma_cq =librosa.feature.chroma_cqt(y=y, sr=sr)
# Computes the chroma variant “Chroma Energy Normalized” (CENS):
chroma_cens =librosa.feature.chroma_cens(y=y, sr=sr)
# Filter Bank Coefficients:
filter_bank = librosa.filters.semitone_filterbank() 

melspectrogram.shape, mfccs.shape, chroma_stft.shape, chroma_cq.shape, chroma_cens.shape


fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(20,20))
#fig.suptitle('Feature set is of dog bark', fontsize=15)


# Spectrogram of dog bark:
Sdb=librosa.amplitude_to_db(abs(S)) #from amplitude to decibel value
img1 = librosa.display.specshow(Sdb, y_axis='log', x_axis='time', ax=ax1) # y_axis = 'log' up there y_axis='linear': Linear-frequency
ax1.set_title('Power Spectrogram'+'\nClass: '+str(df['class'][0]))
fig.colorbar(img1, ax=ax1, format="%+2.0f dB")

# Mel-spectrogram of dog bark:
img2 = librosa.display.specshow(librosa.power_to_db(melspectrogram,ref=np.max),
                               y_axis='mel', fmax=8000,x_axis='time', ax=ax2)
ax2.set_title('Mel Spectrogram'+'\nClass: '+str(df['class'][0]))
fig.colorbar(img2, ax=ax2, format="%+2.0f dB")

# Chromagram of dog bark:
img3 = librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time', ax=ax3)
ax3.set_title('Chromagram'+'\nClass: '+str(df['class'][0]))
fig.colorbar(img3, ax=ax3)

# Chroma cqt of a dog bark:
img4 = librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time', ax=ax4)
ax4.set_title('Chroma_cqt'+'\nClass: '+str(df['class'][0]))
fig.colorbar(img4, ax=ax4)

# Chroma cens of a dog bark:
img5 = librosa.display.specshow(chroma_cens, y_axis='chroma', x_axis='time', ax=ax5)
ax5.set_title('Chroma_cens'+'\nClass: '+str(df['class'][0]))
fig.colorbar(img5, ax=ax5)

plt.savefig("feature_set")
plt.legend()
plt.tight_layout()

# Lets read with scipy:

file1 = df.path_file[1]
class_label = df['class'][1]

wave_sample_rate, wave_audio = wav.read(file1) 


#wave_audio.shape # 2 channels, 176400Hz sampling rate

def wave_plot(filename,class_label):
    rate, wav_audio = wav.read(filename)
    wave_file = open(filename,"rb")
    
    riff_fmt = wave_file.read(36)
    bit_depth_string = riff_fmt[-2:]
    bit_depth = struct.unpack("H",bit_depth_string)[0]
    
    print(f"Sampling Rate: {rate}Hz\nBit Depth: {bit_depth}\nNumber of Channels: {wav_audio.shape[1]}\nDuration: {wav_audio.shape[0]/rate}second\nNumber of Samples: {len(wav_audio)}\nClass: {class_label}")
    
    plt.figure(figsize=(12, 4))
    plt.savefig("wave")
    plt.plot(wav_audio) 
    
    return ipd.Audio(filename)

wave_plot(file1,class_label)

def path_class(filename):
    excerpt = metadata[metadata['path_file'] == filename]
    path_name = os.path.join(path_, 'fold'+str(excerpt.fold.values[0]), filename)
    
    return path_name, excerpt['class'].values[0]

def wav_fmt_parser(filename):
    full_path, _ = path_class(filename)
    wave_file = open(filename,"rb")
    
    riff_fmt = wave_file.read(36)
    
    n_channels_string = riff_fmt[22:24]
    n_channels = struct.unpack("H",n_channels_string)[0]
    
    s_rate_string = riff_fmt[24:28]
    s_rate = struct.unpack("I",s_rate_string)[0]
    
    bit_depth_string = riff_fmt[-2:]
    bit_depth = struct.unpack("H",bit_depth_string)[0]
    
    return (n_channels,s_rate,bit_depth)


audiodf = metadata.copy()
wav_fmt_data = [wav_fmt_parser(i) for i in audiodf['path_file']]
audiodf[['n_channels','sampling_rate','bit_depth']] = pd.DataFrame(wav_fmt_data)

audiodf.to_csv('./audiodf.csv')
audiodf.head()

audiodf.sampling_rate.value_counts()

audiodf.sampling_rate.mean()

audiodf.n_channels.value_counts()

mono = list(audiodf[audiodf.n_channels.values==1]['classID'].value_counts().sort_index())
stereo = list(audiodf[audiodf.n_channels.values==2]['classID'].value_counts().sort_index())
dff = pd.DataFrame([mono,stereo],index=['mono1','stereo2'], columns=classes)
fig, ax = plt.subplots(figsize=(8,6))
dff.T.plot(kind = 'barh', ax = ax,
                    stacked = True, 
                    width=0.7, 
                    bottom=100, 
                    align='center', 
                    color=['lightsteelblue', 
                           'darkred'])

sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
sns.despine()
plt.savefig("mono_stereo")
plt.show()



import torchaudio
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

label1 = metadata['classID'][1]

waveform, sr = torchaudio.load(file1)
audio_mono = torch.mean(waveform, dim=0, keepdim=True)



waveform, sr = torchaudio.load(file1) # load audio
audio_mono = torch.mean(waveform, dim=0, keepdim=True) # Convert sterio to mono
tempData = torch.zeros([1, 160000])
if audio_mono.numel() < 160000: # if sample_rate < 160000
    tempData[:, :audio_mono.numel()] = audio_mono
else:
    tempData = audio_mono[:, :160000] # else sample_rate 160000
audio_mono=tempData



mel_specgram = torchaudio.transforms.MelSpectrogram(sr)(audio_mono)
mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std() # Noramalization
mfcc = torchaudio.transforms.MFCC(sample_rate=sr)(audio_mono) # (channel, n_mfcc, time)
mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std() # mfcc norm
new_feat = torch.cat([mel_specgram, mfcc], axis=1)


 { "specgram": torch.tensor(new_feat[0].permute(1, 0), dtype=torch.float),
    "label": torch.tensor(label1, dtype=torch.long) }
    
    
#Calculating mfcc for the audio data named as file:

audio_data, sr = librosa.load(file1)
mfccs1 = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)


max_pad_len = 174
pad_width = max_pad_len - mfccs1.shape[1]
mfccs = np.pad(mfccs1, pad_width=((0, 0), (0, pad_width)), mode='constant')

audio_data, sr = librosa.load(file0)
mfccs0 = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)


max_pad_len = 174
pad_width = max_pad_len - mfccs0.shape[1]
mfccs = np.pad(mfccs0, pad_width=((0, 0), (0, pad_width)), mode='constant')


# Center MFCC coefficient dimensions to the mean:

mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40).T,axis=0)

mfcc_norm = (mfccs1 - mfccs1.mean()) / mfccs1.std()


max_pad_len = 174

def extract_features(file_name):
   
    try:
        # Here kaiser_fast is a technique used for faster extraction:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        # feature scaling:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccs

features = []

# Iterate through each sound file and extract the features: 
for index, row in audiodf.iterrows():
    
    file_name = os.path.join(os.path.abspath(path_),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    
    class_label = row["classID"]
    data = extract_features(file_name)
    
    features.append([data, class_label])

# Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])


featuresdf.to_csv('./featuresdf.csv')


# Split the dataset into independent and dependent dataset:
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())


# Encode the classification labels
le = LabelEncoder()
y = to_categorical(le.fit_transform(y)) 

# Split the dataset:
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.8, random_state = 3)
X_val, X_test, y_val, y_test =  train_test_split(X_temp, y_temp, train_size=0.5, random_state = 3)


x_train1 = X_train 
x_test1 = X_test
y_train1 = y_train
y_test1 = y_test
x_val1 = X_val
y_val1 = y_val

# Reshaping into 2d to save in csv format:
def dim_2d_save(x_train,x_test,y_train,y_test,x_val,y_val):
    x_train_2d = np.reshape(X_train,(X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
    x_test_2d = np.reshape(X_test,(X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
    x_val_2d = np.reshape(X_val,(X_val.shape[0],X_val.shape[1]*X_val.shape[2]))
    # Saving the data numpy arrays:
    np.savetxt("train_data.csv", x_train_2d, delimiter=",")
    np.savetxt("test_data.csv",x_test_2d,delimiter=",")
    np.savetxt("train_labels.csv",y_train,delimiter=",")
    np.savetxt("test_labels.csv",y_test,delimiter=",")
    np.savetxt("x_val.csv",x_val_2d,delimiter=",")
    np.savetxt("y_val.csv",y_val,delimiter=",")
    return x_train_2d.shape,x_test_2d.shape,x_val_2d.shape

dim_2d_save(x_train1,x_test1,y_train1,y_test1,x_val1,y_val1)


print(f"Lenght of the dataset: {len(X)}\nLength of the training dataset: {len(X_train)}\nLenght of the validation dataset: {len(X_val)}\nLenght of the test dataset: {len(X_test)}\n")

print(f"Shape of X Train: {X_train.shape}\nShape of y Train: {y_train.shape}\nShape of X Test: {X_test.shape}\nShape of y Test: {y_test.shape}")



num_rows = 40
num_columns = 174
num_channels = 1

X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns, num_channels)
X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns, num_channels)
print(X_train.shape)

num_labels = y.shape[1]
filter_size = 3

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# this is our neural network model


model_relu = Sequential()
model_relu.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model_relu.add(MaxPooling2D(pool_size=(2,2)))
model_relu.add(Dropout(0.2))

model_relu.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
model_relu.add(MaxPooling2D(pool_size=(2,2)))
model_relu.add(Dropout(0.2))

model_relu.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
model_relu.add(MaxPooling2D(pool_size=(2,2)))
model_relu.add(Dropout(0.2))

model_relu.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
model_relu.add(MaxPooling2D(pool_size=(2,2)))
model_relu.add(Dropout(0.2))

model_relu.add(GlobalAveragePooling2D())
model_relu.add(Flatten())
model_relu.add(Dense(num_labels, activation='softmax'))

model_relu.compile(optimizer='adam', loss='categorical_crossentropy',
                   metrics=['accuracy'])

    

    
model_relu.summary()

# Calculate pre-training accuracy: 
score = model_relu.evaluate(X_test, y_test, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)

num_epochs = 100
num_batch_size = 256

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

history_relu = model_relu.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data = (X_test, y_test), callbacks=[checkpointer], verbose=1)

duration = datetime.now() - start
print("Training completed in time: ", duration)


# Evaluating the model on the training and testing set:
score = model_relu.evaluate(X_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model_relu.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])

# Plotting Loss of CNN 2D - ReLu Model:
metrics = history_relu.history

plt.plot(history_relu.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['train_loss', 'test_loss'])

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()
plt.grid(True)
plt.savefig("loss")
plt.show()


# Plotting Accuracy of CNN 2D - ReLu Model:
plt.plot(metrics['accuracy'], label='train_accuracy')
plt.plot(metrics['val_accuracy'], label='test_accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()
plt.grid(True)
plt.savefig("accuracy")
plt.show()

test_result = model_relu.test_on_batch(X_test, y_test)
print(test_result)

def prediction_(path_file):
    data_sound = extract_features(path_file)
    X = np.array(data_sound)
    pred_result = model_relu.predict(X.reshape(1,40,174,1))
    pred_class = pred_result.argmax()
    pred_prob = pred_result.max()
    print(f"This belongs to class {pred_class} : {classes[int(pred_class)]}  with {pred_prob} probility %")
prediction_(metadata.path_file[500])
ipd.Audio(metadata.path_file[500])
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19 # VGG19
vggModel = VGG19(weights="imagenet", include_top=False, input_shape=Input(shape=(40, 174, 1)))
outputs = vggModel.output
outputs = Flatten(name="flatten")(outputs)
outputs = Dropout(0.5)(outputs)
outputs = Dense(3, activation="softmax")(outputs)
model = Model(inputs=vggModel.input, outputs=outputs)
for layer in vggModel.layers[:-1]:
    layer.trainable = False
optimizer = Adam(lr=0.0001, decay=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_vgg19 = model.fit(train_generator, validation_data=validation_generator, epochs=100, verbose=1, callbacks=[callback])
arr = np.array(metadata["slice_file_name"])
fold = np.array(metadata["fold"])
cla = np.array(metadata["classID"])


for i in range(192, 197, 2):
    path = path_+'fold' + str(fold[i]) + '/' + arr[i]

    prediction_(path)
    y, sr = librosa.load(path)
    plt.figure(figsize=(15,4))
    plt.title('Class: '+classes[int(cla[i])])
    librosa.display.waveshow(y, sr=sr)
    ipd.Audio(path)
