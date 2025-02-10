# Multimodal Hate Speech Detection in Dravidian Languages

## Overview
This repository contains the code and data for our participation in the Shared Task on Multimodal Hate Speech Detection in Dravidian languages: DravidianLangTech@NAACL 2025. The task involves detecting hate speech in Malayalam, Tamil, and Telugu using both text and audio data.

## Shared Task Details
The Multimodal Social Media Data Analysis (MSMDA) shared task is an exciting challenge that invites researchers and practitioners from various disciplines to explore the realm of analyzing complex social media data using a multimodal approach. The MSMDA-DL shared task presents an exciting opportunity for researchers to explore the intricacies of analyzing multimodal content from social media platforms in the rich and diverse linguistic landscape of the Dravidian languages.

### Subtasks
1. **Task 1: Multimodal hate-speech detection in Malayalam**
2. **Task 2: Multimodal hate-speech detection in Tamil**
3. **Task 3: Multimodal hate-speech detection in Telugu**

Participants were provided with training data and required to develop models that can analyze the text and speech components of the data and predict the respective labels. The evaluation was based on the macro-F1 score.

## Our Results
We participated in all three tasks and achieved the following rankings based on the micro F1 score:
1. **Tamil**: 3rd place (F1-score = 0.6438)
2. **Telugu**: 15th place (F1-score = 0.1559)
3. **Malayalam**: 12th place (F1-score = 0.3016)

## Methodologies

### Preprocessing Techniques
1. **Text Preprocessing**:
   - **Normalization**: Using `IndicNormalizerFactory` to normalize text.
   - **Tokenization**: Tokenizing text using `indic_tokenize`.
   - **Stopword Removal**: Removing stopwords using `advertools`.

2. **Audio Preprocessing**:
   - **Feature Extraction**: Extracting MFCC features using `librosa`.
   - **Normalization**: Applying TTS-like normalization and dynamic audio normalization.

### Models
1. **Text Models**:
   - **BERT-based Models**: Using `bert-base-multilingual-cased` and `xlm-roberta-large` for text embeddings.
   - **Neural Networks**: Using Sequential models with Dense, Dropout, and BatchNormalization layers.

2. **Audio Models**:
   - **CNN-based Models**: Using Convolutional Neural Networks (CNN) for audio classification.

### Data Augmentation
- **Text Augmentation**: Using `nlpaug` for synonym augmentation.
- **Audio Augmentation**: Adding noise, time-stretching, and pitch-shifting.

## Model Comparison

### Text Models
| Model                     | Language  | Accuracy  |
|---------------------------|-----------|-----------|
| TF-IDF (Term Frequency Inverse Document Frequency) | Tamil     | 0.55      |
| Count Vectorizer | Tamil     | 0.59      |
| BERT (bert-base-multilingual-cased) | Tamil     | 0.67      |
| XLM-RoBERTa (xlm-roberta-base)     | Tamil     | 0.73      |
| XLM-RoBERTa (xlm-roberta-large)     | Tamil     | 0.83      |
| TF-IDF (Term Frequency Inverse Document Frequency) | Telugu     | 0.58      |
| Count Vectorizer | Telugu     | 0.60      |
| BERT (bert-base-multilingual-cased) | Telugu    | 0.71      |
| XLM-RoBERTa (xlm-roberta-base)     | Telugu    | 0.71      |
| XLM-RoBERTa (xlm-roberta-large)     | Telugu    | 0.88      |
| TF-IDF (Term Frequency Inverse Document Frequency) | Malayalam     | 0.58      |
| Count Vectorizer | Malayalam     | 0.58      |
| BERT (bert-base-multilingual-cased) | Malayalam | 0.73      |
| XLM-RoBERTa (xlm-roberta-base)     | Malayalam | 0.72      |
| XLM-RoBERTa (xlm-roberta-large)     | Malayalam | 0.85      |

### Audio Models
| Model                     | Language  | Accuracy  |
|---------------------------|-----------|-----------|
| CNN (without normalization) | Tamil     | 0.64      |
| CNN (with normalization)    | Tamil     | 0.58      |
| CNN (with augmentation)     | Tamil     | 0.88      |
| CNN (without normalization) | Telugu    | 0.54      |
| CNN (with normalization)    | Telugu    | 0.54      |
| CNN (with augmentation)     | Telugu    | 0.88      |
| CNN (without normalization) | Malayalam | 0.85      |
| CNN (with normalization)    | Malayalam | 0.80      |
| CNN (with augmentation)     | Malayalam | 0.93      |

## Dataset
The dataset contains hate speech utterances sourced from YouTube videos for Malayalam, Tamil, and Telugu languages. The dataset has the following classes:

| Language  | Class       | Subclasses                                      |
|-----------|-------------|-------------------------------------------------|
| Malayalam | Hate        | Gender (G), Political (P), Religious (R), Personal Defamation (C) |
|           | Non-Hate    | NH                                              |
| Tamil     | Hate        | Gender (G), Political (P), Religious (R), Personal Defamation (C) |
|           | Non-Hate    | NH                                              |
| Telugu    | Hate        | Gender (G), Political (P), Religious (R), Personal Defamation (C) |
|           | Non-Hate    | NH                                              |

### File Nomenclature
Files in the dataset follow a specific naming convention:
`Binary_Class_Language_SubjectID_ClassLabel_Gender_Class_Main_Number_SubNumber.WAV`

Example:
For the file named `H_ML_001_C_F_044_001.WAV`:
- `H`: Indicates Hate speech.
- `ML`: Malayalam language.
- `001`: Subject ID.
- `C`: Class Label (Personal Defamation).
- `F`: Female speaker.
- `044`: Main Number (YouTube link number).
- `001`: Sub Number (utterance number).

### Folder Structure
- Each language subfolder (Malayalam, Tamil, Telugu) contains:
  - **Audio**: Audio files of the utterances.
  - **Text**: Text transcripts of the audio utterances, along with file names and class labels.

## Authors
- Vasantharan K (vasantharank.work@gmail.com)
- Prethish GA (prethish0409@gmail.com)