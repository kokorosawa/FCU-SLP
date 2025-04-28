from scipy.io import wavfile
from scipy.signal import spectrogram
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import resample_poly
from scipy.io.wavfile import write
import torch
import torchaudio.functional as F
from python_speech_features import mfcc


def preprocess_pipeline(file_path, spk:str, target_fs=16000):
    fs, wav = wavfile.read(file_path)
    if fs != target_fs:
        wav = resample(wav, fs, target_fs)

    wav = to_mono(wav)
    wav = vad(wav, target_fs)
    wav = extract_mfcc(wav, target_fs)

    # Ensure the array has shape (1300, 13)
    target_shape = (1300, 13)
    if wav.shape[0] < target_shape[0]:
        padding = target_shape[0] - wav.shape[0]
        wav = np.pad(wav, ((0, padding), (0, 0)), mode='constant', constant_values=0)
    elif wav.shape[0] > target_shape[0]:
        wav = wav[:target_shape[0], :]

    label = np.zeros(10)
    match spk:
        case "dsp": label = 0
        case "fyl": label = 1
        case "hym": label = 2
        case "hyy": label = 3
        case "jzy": label = 4
        case "ljw": label = 5
        case "lmz": label = 6
        case "lph": label = 7
        case "sya": label = 8
        case "wjm": label = 9
        
    return wav, label

def resample(wav, fs, target_fs):
    """
    Resample the audio signal to the target sampling frequency.
    """
    wav_resampled = resample_poly(wav, target_fs, fs)
    return wav_resampled

def to_mono(wav):
    """
    Convert stereo audio to mono by averaging the two channels.
    """
    if len(wav.shape) == 2:  # Check if the audio is stereo
        wav = np.mean(wav, axis=1).astype(wav.dtype)
    return wav

def vad(wav, fs):
    """
    Apply Voice Activity Detection (VAD) to the audio signal.
    """
    wav = torch.from_numpy(wav).float()
    vad_wav = F.vad(wav, fs, trigger_level=7) 
    return vad_wav.numpy()

def extract_mfcc(wav, fs, frame_size=256, hop_size=128, num_cepstral=13):
    mfcc_features = mfcc(wav, samplerate=fs, numcep=num_cepstral, winlen=frame_size/fs, winstep=hop_size/fs, appendEnergy=False)
    # print(mfcc_features.shape)
    return mfcc_features




