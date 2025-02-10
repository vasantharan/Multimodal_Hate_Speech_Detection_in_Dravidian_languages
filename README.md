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