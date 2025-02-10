import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
import advertools as adv
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from transformers import AutoTokenizer, AutoModel
import torch
import pickle

stopwords = list(sorted(adv.stopwords['telugu']))
def preprocess_telugu_text(text):
    tokens = list(indic_tokenize.trivial_tokenize(text, lang="te"))
    tokens = [token for token in tokens if token not in stopwords]
    processed_text = ' '.join(tokens)
    return processed_text
def extract_embeddings(model_name, texts):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    embeddings = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded_inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            outputs = model(**encoded_inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            embeddings.extend(batch_embeddings)
    return np.array(embeddings)

def tts_audio_normalizer(audio, target_peak=-1.0, gain_dB=0.0):
    target_amplitude = 10 ** (target_peak / 20)
    current_peak = np.max(np.abs(audio))
    if current_peak > 0:
        normalization_factor = target_amplitude / current_peak
    else:
        normalization_factor = 1.0
    normalized_audio = audio * normalization_factor
    gain_amplitude = 10 ** (gain_dB / 20)
    normalized_audio = normalized_audio * gain_amplitude
    return normalized_audio
def dynamic_audio_normalizer(audio, sr):
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio / peak
    return audio

label_encoder_path = '../label encoder/tel_label_encoder_classes.npy'
text_label_encoder_path = '../label encoder/tel_label_encoder.pkl'
with open(text_label_encoder_path, 'rb') as f:
    text_label_encoder = pickle.load(f)
with open(label_encoder_path, 'rb') as f:
    label_encoder = np.load(f, allow_pickle=True)
audio_model = tf.keras.models.load_model('../models/telugu_best_audio_model.keras')
text_model = tf.keras.models.load_model('../models/best_tel_hyper_classification_model.h5')
feature_extractor = "xlm-roberta-large"

text_result_file = 'the_deathly_hallows_teluguText.tsv'
audio_result_file = 'the_deathly_hallows_teluguAudio.tsv'
text_df = pd.DataFrame()
audio_df = pd.DataFrame()

def predict_audio(file_path, label_encoder=text_label_encoder, audio_model=audio_model):
    SAMPLE_RATE = 22050  
    DURATION = 3  
    MFCC_FEATURES = 40 
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    def extract_features(audio, n_mfcc=MFCC_FEATURES, duration=DURATION, sr=SAMPLE_RATE):
        try:
            normalized_audio = tts_audio_normalizer(audio, target_peak=-1.0, gain_dB=3.0)
            normalized_audio = dynamic_audio_normalizer(normalized_audio, sr)
            mfccs = librosa.feature.mfcc(y=normalized_audio, sr=sr, n_mfcc=n_mfcc)
            return np.mean(mfccs.T, axis=0)
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None
    features = extract_features(audio)
    reshaped_features = features.reshape((1, MFCC_FEATURES, 1, 1))
    predicted_class = np.argmax(audio_model.predict(reshaped_features), axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0]

def load_dataset(base_dir= '../test', lang='telugu'):
    dataset = []
    lang_dir = os.path.join(base_dir, lang)
    audio_dir = os.path.join(lang_dir, "audio")
    text_dir = os.path.join(lang_dir, "text")
    text_file = os.path.join(text_dir, [file for file in os.listdir(text_dir) if file.endswith(".xlsx")][0])
    text_df = pd.read_excel(text_file)
    for file in text_df['File_Name']:
        if (file + ".wav") in os.listdir(audio_dir):
            audio_path = os.path.join(audio_dir, file + ".wav")
            transcript_row = text_df.loc[text_df["File_Name"] == file]
            if not transcript_row.empty:
                transcript = transcript_row.iloc[0]["Transcript"]
                dataset.append({
                    "File_Name": audio_path,
                    "Transcript": transcript,
                })
        else:
            transcript_row = text_df.loc[text_df["File_Name"] == file]
            if not transcript_row.empty:
                transcript = transcript_row.iloc[0]["Transcript"]
                dataset.append({
                    "File_Name": "Nil",
                    "Transcript": transcript,
                })
    return pd.DataFrame(dataset)
dataset_df = load_dataset()
dataset_df['File_Name'] = dataset_df['File_Name'].str.replace('\\', '/', regex=False)

dataset_df['Transcript'] = dataset_df['Transcript'].apply(preprocess_telugu_text)
text_embeddings = extract_embeddings(feature_extractor, dataset_df['Transcript'].tolist())
text_predictions = text_model.predict(text_embeddings)
text_labels = text_label_encoder.inverse_transform(np.argmax(text_predictions, axis=1))

text_df['File_Name'] = dataset_df['File_Name'].str.replace('../test/telugu/audio/', '', regex=False)
text_df['Class Label Short'] = text_labels
text_df.to_csv(text_result_file, sep='\t', index=False)
print('Text Classification Done')

audio_df['File_Name'] = dataset_df['File_Name'].str.replace('../test/telugu/audio/', '', regex=False)
audio_df['Class Label Short'] = dataset_df['File_Name'].apply(predict_audio)
audio_df.to_csv(audio_result_file, sep='\t', index=False)
print('Audio Classification Done')