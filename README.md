# 🚀 Multimodal Hate Speech Detection in Dravidian Languages

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14847234.svg)](https://doi.org/10.5281/zenodo.14847234)

## 📌 Overview
This repository contains the code and data for our participation in the **Shared Task on Multimodal Hate Speech Detection in Dravidian languages: DravidianLangTech@NAACL 2025**. The task involves detecting hate speech in **Malayalam, Tamil, and Telugu** using both **text and audio data**.

---

## 📖 Shared Task Details
The **Multimodal Social Media Data Analysis (MSMDA) shared task** is an exciting challenge that invites researchers and practitioners to explore the complexities of analyzing **multimodal social media data** in the rich and diverse linguistic landscape of **Dravidian languages**.

### 🏆 Subtasks
1. **Task 1**: Multimodal hate-speech detection in **Malayalam** 
2. **Task 2**: Multimodal hate-speech detection in **Tamil** 
3. **Task 3**: Multimodal hate-speech detection in **Telugu** 

📊 **Evaluation Metric**: The models were evaluated based on the **Macro-F1 Score**.

---

## 🏅 Our Results
We participated in all three tasks and achieved the following rankings based on the **Micro-F1 Score**:

| **Language**  | **Ranking** | **F1-Score** |
|--------------|------------|-------------|
| 🏆 **Tamil**    | **3rd Place** | **0.6438** |
| 🎯 **Telugu**   | **15th Place** | **0.1559** |
| 🎯 **Malayalam** | **12th Place** | **0.3016** |

---

## 🛠 Methodologies

### 🔍 **Preprocessing Techniques**
#### **Text Preprocessing**
✔ **Normalization**: Using `IndicNormalizerFactory` to normalize text.  
✔ **Tokenization**: Tokenizing text using `indic_tokenize`.  
✔ **Stopword Removal**: Removing stopwords using `advertools`.

#### **Audio Preprocessing**
✔ **Feature Extraction**: Extracting **MFCC features** using `librosa`.  
✔ **Normalization**: Applying **TTS-like normalization** and **dynamic audio normalization**.

---

### 🤖 **Models Used**
#### **Text Models**
- **BERT-based Models**: `bert-base-multilingual-cased`, `xlm-roberta-large` for text embeddings.
- **Neural Networks**: Sequential models with **Dense, Dropout, and BatchNormalization layers**.

#### **Audio Models**
- **CNN-based Models**: **Convolutional Neural Networks (CNN)** for audio classification.

---

### 🎛 **Data Augmentation**
✔ **Text Augmentation**: Using `nlpaug` for **synonym augmentation**.  
✔ **Audio Augmentation**: Adding **noise, time-stretching, and pitch-shifting**.

---

## 📊 Model Comparison

### 📜 **Text Models**
| **Model**                                   | **Language**  | **Accuracy** |
|---------------------------------------------|--------------|-------------|
| **TF-IDF**                                  | Tamil        | 🔴 **0.55**  |
| **Count Vectorizer**                        | Tamil        | 🔴 **0.59**  |
| **BERT (bert-base-multilingual-cased)**     | Tamil        | 🟡 **0.67**  |
| **XLM-RoBERTa (xlm-roberta-base)**         | Tamil        | 🟢 **0.73**  |
| **XLM-RoBERTa (xlm-roberta-large)**        | Tamil        | 🟢 **0.83**  |
| **TF-IDF**                                  | Telugu       | 🔴 **0.58**  |
| **Count Vectorizer**                        | Telugu       | 🔴 **0.60**  |
| **BERT (bert-base-multilingual-cased)**     | Telugu       | 🟡 **0.71**  |
| **XLM-RoBERTa (xlm-roberta-base)**         | Telugu       | 🟢 **0.71**  |
| **XLM-RoBERTa (xlm-roberta-large)**        | Telugu       | 🟢 **0.88**  |
| **TF-IDF**                                  | Malayalam    | 🔴 **0.58**  |
| **Count Vectorizer**                        | Malayalam    | 🔴 **0.58**  |
| **BERT (bert-base-multilingual-cased)**     | Malayalam    | 🟡 **0.73**  |
| **XLM-RoBERTa (xlm-roberta-base)**         | Malayalam    | 🟢 **0.72**  |
| **XLM-RoBERTa (xlm-roberta-large)**        | Malayalam    | 🟢 **0.85**  |

### 🎵 **Audio Models**
| **Model**                     | **Language**  | **Accuracy**  |
|--------------------------------|--------------|--------------|
| **CNN (without normalization)** | Tamil        | 🟡 **0.64**  |
| **CNN (with normalization)**    | Tamil        | 🔴 **0.58**  |
| **CNN (with augmentation)**     | Tamil        | 🟢 **0.88**  |
| **CNN (without normalization)** | Telugu       | 🔴 **0.54**  |
| **CNN (with normalization)**    | Telugu       | 🔴 **0.54**  |
| **CNN (with augmentation)**     | Telugu       | 🟢 **0.88**  |
| **CNN (without normalization)** | Malayalam    | 🟡 **0.85**  |
| **CNN (with normalization)**    | Malayalam    | 🟡 **0.80**  |
| **CNN (with augmentation)**     | Malayalam    | 🟢 **0.93**  |

---

## 📂 Dataset
The dataset consists of **hate speech utterances** sourced from **YouTube videos** for **Malayalam, Tamil, and Telugu**.

### 🎯 **Dataset Classes**
| **Language**  | **Class**       | **Subclasses**                                      |
|--------------|-------------|-------------------------------------------------|
| Malayalam   | **Hate**    | Gender (G), Political (P), Religious (R), Personal Defamation (C) |
|             | **Non-Hate**| NH                                              |
| Tamil       | **Hate**    | Gender (G), Political (P), Religious (R), Personal Defamation (C) |
|             | **Non-Hate**| NH                                              |
| Telugu      | **Hate**    | Gender (G), Political (P), Religious (R), Personal Defamation (C) |
|             | **Non-Hate**| NH                                              |

### 📌 **File Nomenclature**
Naming convention:  
`Binary_Class_Language_SubjectID_ClassLabel_Gender_Class_Main_Number_SubNumber.WAV`

#### **Example**: `H_ML_001_C_F_044_001.WAV`
- `H`: Hate speech
- `ML`: Malayalam
- `C`: Personal Defamation
- `F`: Female speaker

---

### 📌 **DOI Generation**
The dataset and code are archived and assigned a **DOI** through **Zenodo** for proper citation and reference.

---

## 👥 Authors
- **Vasantharan K** ([📧 Email](mailto:vasantharank.work@gmail.com))
- **Prethish GA** ([📧 Email](mailto:prethish0409@gmail.com))

---

📌 **Feel free to contribute!** 🚀
