import cv2
import emoji
import gc
import gensim
import gensim.downloader
import json
import matplotlib.pyplot as plt
import nlp
import nltk
import numpy as np
import os
import pandas as pd
import pathlib
import pattern
import pickle
import qalsadi.lemmatizer
import random
import re
import seaborn as sns
import shutil
import sklearn
import spacy
import string
import tempfile
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import tensorflow_hub as hub
import tensorflow_text
import textblob
import torch
import transformers
import treetaggerwrapper as ttpw
import warnings
import xgboost
from collections import Counter
from gensim.models import KeyedVectors, Word2Vec
from IPython import display
from nltk.corpus import stopwords, stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.isri import ISRIStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from pathlib import Path
from pattern.en import lemma, lexeme
from sklearn import decomposition, ensemble
from sklearn.datasets import load_files
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import (
  accuracy_score, classification_report, confusion_matrix, f1_score,
  precision_recall_fscore_support, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import *
from tensorflow.keras.utils import to_categorical
from textblob import TextBlob, Word
from torch.utils.data import DataLoader, Dataset
from transformers import (
  AutoModel, AutoModelForSequenceClassification, AutoTokenizer, BertConfig, BertTokenizer,
  BertTokenizerFast, TFBertModel, Trainer, TrainingArguments,
)

logdir = pathlib.Path(tempfile.mkdtemp()) / "tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def ShowConfusionMatrix(yTrue, yPred, classes):
  cm = confusion_matrix(yTrue, yPred, normalize='true')
  plt.figure(figsize=(8, 8))
  sp = plt.subplot(1, 1, 1)
  ctx = sp.matshow(cm)
  noOfClasses = len(classes)
  plt.xticks(list(range(noOfClasses)), labels=classes)
  plt.yticks(list(range(noOfClasses)), labels=classes)
  plt.colorbar(ctx)
  plt.show()


def GetPaddedSequences(tokenizer, data, maxLength=50, truncating='post', padding='post'):
  sequences = tokenizer.texts_to_sequences(data)
  padded = pad_sequences(
    sequences,
    truncating=truncating,
    padding=padding,
    maxlen=maxLength
  )
  return padded


def TrainEvaluateModel(
  trainX, trainY, valX, valY, moduleURL, embedSize, name,
  noOfClasses, epochs=32, batchSize=32, trainable=False
):
  tf.keras.backend.clear_session()

  hubLayer = hub.KerasLayer(
    moduleURL, input_shape=[],
    output_shape=[embedSize],
    dtype=tf.string,
    trainable=trainable,
  )
  model = Sequential(
    [
      # InputLayer(input_shape=[], dtype=tf.string),
      # hub.KerasLayer(moduleURL, output_shape=[embedSize], trainable=trainable),
      hubLayer,
      # Dense(128, activation='relu'), # 256
      Dropout(0.25),
      Dense(64, activation='relu'),
      Dropout(0.5),
      Dense(64, activation='relu'),
      Dense(noOfClasses, activation='sigmoid'),
    ]
  )
  optimizer = Adam()  # (learning_rate=5e-4)

  callbacks = [
    # EarlyStopping(monitor='val_loss', patience=10, mode='min'),
    tfdocs.modeling.EpochDots(),
    # TensorBoard(logdir/name),
    ModelCheckpoint("model.h5", monitor="val_accuracy", save_best_only=True),
  ]
  model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
      'accuracy',
      TruePositives(name="TP"),
      TrueNegatives(name="TN"),
      FalsePositives(name="FP"),
      FalseNegatives(name="FN"),
    ]
  )
  history = model.fit(
    trainX, trainY,
    validation_data=(valX, valY),
    epochs=epochs,
    batch_size=batchSize,
    callbacks=callbacks,
    verbose=0
  )
  return history, model


class TweetDataset(Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)


def TorchComputeMetrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  acc = accuracy_score(labels, preds)
  tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
  epsilon = np.finfo(float).eps
  precision = tp / float(tp + fp + epsilon)
  recall = tp / float(tp + fn + epsilon)
  specificity = tn / float(tn + fp + epsilon)
  f1 = (2.0 * tp) / float(2.0 * tp + fn + fp + epsilon)
  return {
    'accuracy'   : acc,
    'f1'         : f1,
    'precision'  : precision,
    'recall'     : recall,
    'specificity': specificity,
    'TP'         : tp,
    'TN'         : tn,
    'FP'         : fp,
    'FN'         : fn,
  }


def ModelBiLSTM1():
  tf.keras.backend.clear_session()

  inp = Input(shape=(maxLength,))
  x = Embedding(noOfWords, embedSize, input_length=maxLength)(inp)
  x = Bidirectional(LSTM(512, return_sequences=True))(x)
  avgPool = GlobalAveragePooling1D()(x)
  maxPool = GlobalMaxPooling1D()(x)
  conc = concatenate([avgPool, maxPool])
  conc1 = Dense(
    128,
    activation="relu",
    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=l2(1e-4),
    activity_regularizer=l2(1e-5),
  )(conc)
  conc2 = Dense(
    64,
    activation="relu",
    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=l2(1e-4),
    activity_regularizer=l2(1e-5),
  )(conc)
  conc3 = Dense(
    32,
    activation="relu",
    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=l2(1e-4),
    activity_regularizer=l2(1e-5),
  )(conc)
  conc1x = Dense(256, activation="relu")(conc)
  conc2x = Dense(128, activation="relu")(conc)
  conc3x = Dense(64, activation="relu")(conc)
  conc = concatenate([conc1, conc2, conc3, conc1x, conc2x, conc3x])
  conc = Dropout(0.5)(conc)
  conc = Dense(64, activation="relu")(conc)
  conc = Dropout(0.25)(conc)
  conc = Dense(64, activation="relu")(conc)
  conc = Dropout(0.25)(conc)
  conc = Dense(64, activation="relu")(conc)
  conc = Dropout(0.25)(conc)
  outp = Dense(1, activation="sigmoid")(conc)
  model = Model(inputs=inp, outputs=outp)

  callbacks = [
    EarlyStopping(monitor='val_loss', patience=25, mode='min'),
    tfdocs.modeling.EpochDots(),
    # TensorBoard(logdir/name),
    ModelCheckpoint("model.h5", monitor="val_accuracy", save_best_only=True, verbose=1),
  ]

  metrics = [
    'accuracy',
    TruePositives(name="TP"),
    TrueNegatives(name="TN"),
    FalsePositives(name="FP"),
    FalseNegatives(name="FN"),
  ]

  model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.00001),
    metrics=metrics,
  )
  model.summary()

  history = model.fit(
    trainPaddedSequences, trainLabels,
    validation_data=(validationPaddedSequences, validationLabels),
    epochs=epochs * 10,
    batch_size=batchSize,
    callbacks=callbacks,
  )
  return model, history


def ModelCNN1():
  tf.keras.backend.clear_session()

  model = Sequential(
    [
      Embedding(noOfWords, 128, input_length=maxLength),
      Conv1D(32, 5, activation='relu'),
      MaxPooling1D(),
      Conv1D(64, 5, activation='relu'),
      MaxPooling1D(),
      Conv1D(128, 5, activation='relu'),
      MaxPooling1D(),
      Flatten(),
      Dense(32, activation='relu'),
      Dropout(0.5),
      Dense(64, activation='relu'),
      Dropout(0.5),
      Dense(noOfClasses, activation='softmax'),
    ]
  )
  model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy'],
  )
  model.summary()

  history = model.fit(
    trainPaddedSequences, trainLabelsCat,
    validation_data=(validationPaddedSequences, validationLabelsCat),
    epochs=epochs,
    batch_size=batchSize,
    callbacks=[
      EarlyStopping(monitor='val_accuracy', patience=25),
      ModelCheckpoint("model.hdf5", monitor='val_loss'),
    ]
  )
  return model, history


maxLength = 100
noOfWords = 2500
epochs = 25
batchSize = 32
trainSize = 0.85
embedSize = 300
overwrite = False
# SEQ_MAX_LEN = 100 #'auto'

workingPath = "/content/drive/MyDrive/Suicide Research"
dataPath = "/content/drive/MyDrive/Suicide Research/انتحار_filtered_tweets.xlsx"
documentsPath = os.path.join(workingPath, "documents_tweets_annotators.p")

if (not overwrite and os.path.exists(documentsPath)):
  allDocuments, classes, classesCat = FileHelper().LoadPickle(documentsPath)
  noOfClasses = len(set(classes))
else:
  ar = ArabicTextHelper()
  tweets = pd.read_excel(dataPath, sheet_name='tweets_annotator_1')['tweet'].values

  # Five annotators:
  classes1 = pd.read_excel(dataPath, sheet_name='tweets_annotator_1')['category'].values
  classes2 = pd.read_excel(dataPath, sheet_name='tweets_annotator_2')['category'].values
  classes3 = pd.read_excel(dataPath, sheet_name='tweets_annotator_3')['category'].values
  classes4 = pd.read_excel(dataPath, sheet_name='tweets_annotator_4')['category'].values
  classes5 = pd.read_excel(dataPath, sheet_name='tweets_annotator_5')['category'].values

  classes1 = np.array(classes1).reshape(-1, 1)
  classes2 = np.array(classes2).reshape(-1, 1)
  classes3 = np.array(classes3).reshape(-1, 1)
  classes4 = np.array(classes4).reshape(-1, 1)
  classes5 = np.array(classes5).reshape(-1, 1)

  classesStack = np.hstack((classes1, classes2, classes3, classes4, classes5))
  classesMod = [[int(el) for el in cls if el == el] for cls in classesStack]
  print("Lengths:", [len(arr) for arr in classesMod])
  classes = [Counter(el).most_common(1)[0][0] for el in classesMod]

  noOfClasses = len(set(classes))
  classesCat = to_categorical(classes, num_classes=noOfClasses)
  classes = ['Suicide' if cls == 1 else 'Normal' for cls in classes]
  preTweets = ar.ArabicRegexPreprocessing(tweets)
  tweetsLemmaISRI = ar.ArabicISRIStemmerPreprocessing(preTweets)
  tweetsLemmaQalsadi = ar.ArabicQalsadiLemmatizerPreprocessing(preTweets)
  tweetsStopISRI = ar.ArabicStopwordsRemovalPreprocessing(tweetsLemmaISRI)
  tweetsStopQalsadi = ar.ArabicStopwordsRemovalPreprocessing(tweetsLemmaQalsadi)
  allDocuments = [
    tweets,
    preTweets,
    tweetsLemmaISRI,
    tweetsLemmaQalsadi,
    tweetsStopISRI,
    tweetsStopQalsadi,
  ]
  FileHelper().StorePickle((allDocuments, classes, classesCat), documentsPath)

for j in [40]:
  for i in range(len(allDocuments)):
    print(allDocuments[i][j])
    print(type(allDocuments[i]))
  print(classes[j])
  print(classesCat[j])

print(len([1 for i in range(len(allDocuments[0])) if classes[i] == 'Normal']))
print(len([1 for i in range(len(allDocuments[0])) if classes[i] != 'Normal']))

stopwordsList = stopwords.words('arabic')
print(len(stopwordsList))
print(stopwordsList)

tf.keras.backend.clear_session()
histories = {}
evalResults = {}

for i in range(0, len(allDocuments)):
  uniqueClasses = set(classes)
  noOfClasses = 1  # len(uniqueClasses)
  labelEncoder = LabelEncoder()
  classes = labelEncoder.fit_transform(classes)
  trainTweets, testTweets, trainLabels, testLabels = train_test_split(
    np.array(allDocuments[i]), classes, train_size=trainSize, stratify=classes
  )
  trainTweets, validationTweets, trainLabels, validationLabels = train_test_split(
    trainTweets, trainLabels, train_size=trainSize, stratify=trainLabels
  )

  moduleURLs = {
    # https://tfhub.dev/google/universal-sentence-encoder-multilingual/3
    # "universal-sentence-encoder-multilingual-256": ["https://tfhub.dev/google/universal-sentence-encoder-multilingual/3", 256], # Multilingual
    # https://tfhub.dev/google/universal-sentence-encoder-multilingual/3
    # "universal-sentence-encoder-multilingual-large-256": ["https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3", 256], # Multilingual
    # https://tfhub.dev/google/universal-sentence-encoder-multilingual/3
    # "universal-sentence-encoder-multilingual-qa-256": ["https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3", 256], # Multilingual
  }

  for key in moduleURLs.keys():
    moduleURL, embedSize = moduleURLs[key]
    histories[key + "-finetuned-" + str(i + 1)], model = TrainEvaluateModel(
      trainTweets, trainLabels,
      validationTweets, validationLabels,
      moduleURL,
      embedSize,
      key,
      noOfClasses=noOfClasses,
      epochs=epochs,
      batchSize=batchSize,
      trainable=True,
    )
    model.load_weights("model.h5")
    evalResults[key + "-finetuned-" + str(i + 1)] = model.evaluate(testTweets, testLabels)
    print(key + "-finetuned-" + str(i + 1))
    print(evalResults[key + "-finetuned-" + str(i + 1)])

  for key in moduleURLs.keys():
    moduleURL, embedSize = moduleURLs[key]
    histories[key + "-" + str(i + 1)], model = TrainEvaluateModel(
      trainTweets, trainLabels,
      validationTweets, validationLabels,
      moduleURL,
      embedSize,
      key,
      noOfClasses=noOfClasses,
      epochs=epochs,
      batchSize=batchSize,
      trainable=False,
    )
    evalResults[key + "-" + str(i + 1)] = model.evaluate(testTweets, testLabels)
    print(key + "-" + str(i + 1))
    print(evalResults[key + "-" + str(i + 1)])

plt.rcParams['figure.figsize'] = (12, 8)
plotter = tfdocs.plots.HistoryPlotter(metric='accuracy')
plotter.plot(histories)
plt.xlabel("Epochs")
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.title("Accuracy Curves for Models")
plt.show()

plotter = tfdocs.plots.HistoryPlotter(metric='loss')
plotter.plot(histories)
plt.xlabel("Epochs")
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.title("Loss Curves for Models")
plt.show()

print(evalResults)

BERT_MODEL_NAME = 'aubmindlab/bert-base-arabertv2'
keyword = 'bert-base-arabertv2'

for i in range(len(allDocuments)):
  torch.cuda.empty_cache()
  gc.collect()

  outputDir = os.path.join(workingPath, "Experiments", f"{keyword}-{i + 1}")

  bert = AutoModel.from_pretrained(BERT_MODEL_NAME)
  tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
  model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=noOfClasses)

  tweets = list(allDocuments[i])
  trainTweets, testTweets, trainLabels, testLabels = train_test_split(
    tweets, np.argmax(classesCat, axis=-1), train_size=trainSize, random_state=0, stratify=classes
  )
  trainTweets, validationTweets, trainLabels, validationLabels = train_test_split(
    trainTweets, trainLabels, train_size=trainSize, random_state=0, stratify=trainLabels
  )

  maxTweetLength = max(len(tweet) for tweet in tweets)

  allEncodings = tokenizer(tweets, truncation=True, padding=True, max_length=maxLength)
  trainEncodings = tokenizer(trainTweets, truncation=True, padding=True, max_length=maxLength)
  validationEncodings = tokenizer(validationTweets, truncation=True, padding=True, max_length=maxLength)
  testEncodings = tokenizer(testTweets, truncation=True, padding=True, max_length=maxLength)

  # allEncodings = TweetDataset(allEncodings, classesCat)
  allEncodings = TweetDataset(allEncodings, np.argmax(classesCat, axis=-1))
  trainDataset = TweetDataset(trainEncodings, trainLabels)
  validationDataset = TweetDataset(validationEncodings, validationLabels)
  testDataset = TweetDataset(testEncodings, testLabels)

  print("Max tweet length:", maxTweetLength)
  print("Train subset length:", len(trainTweets))
  print("Test subset length:", len(testTweets))
  print("Validation subset length:", len(validationTweets))
  print("Train labels length:", len(trainLabels))
  print("Test labels length:", len(testLabels))
  print("Validation labels length:", len(validationLabels))
  print("No. of the train tweets:", len(trainTweets))
  print(f"The first train tweet: ({trainTweets[0]}) with a label ({trainLabels[0]}).")
  print("The first train label:", trainLabels[0])
  print("No. of the validation tweets:", len(validationTweets))
  print(f"The first validation tweet: ({validationTweets[0]}) with a label ({validationLabels[0]}).")
  print("The first validation label:", validationLabels[0])
  print("No. of the test tweets:", len(testTweets))
  print(f"The first test tweet: ({testTweets[0]}) with a label ({testLabels[0]}).")
  print("The first test label:", testLabels[0])
  print("trainEncodings[0]:", trainEncodings[0])
  print("validationEncodings[1]:", validationEncodings[0])
  print("testEncodings[2]:", testEncodings[0])

  trainingArgs = TrainingArguments(
    output_dir=outputDir,  # The output directory.
    num_train_epochs=epochs,  # The total number of training epochs.
    per_device_train_batch_size=8,  # The batch size per device during training.
    per_device_eval_batch_size=16,  # The batch size for evaluation.
    warmup_steps=1500,  # The number of warmup steps for learning rate scheduler.
    weight_decay=0.01,  # The strength of weight decay.
    logging_strategy='epoch',
    evaluation_strategy='epoch',
    # save_strategy='epoch',
    # load_best_model_at_end=True,
    # logging_dir=outputDir+"/logs",
    # save_total_limit=1,
  )

  trainer = Trainer(
    model=model,  # The instantiated 🤗 Transformers model to be trained.
    args=trainingArgs,  # The training arguments, defined above.
    train_dataset=trainDataset,  # The training dataset.
    eval_dataset=validationDataset,  # The evaluation dataset.
    compute_metrics=TorchComputeMetrics,
  )

  trainer.train()
  trainer.save_model(outputDir + "/best")
  print(i + 1, trainer.evaluate(testDataset))
  print(i + 1, trainer.evaluate(allEncodings))

  del trainer
  del trainDataset
  del validationDataset
  del testDataset
  del allEncodings
  del trainingArgs

  gc.collect()

trainTweets, testTweets, trainLabels, testLabels = train_test_split(
  allDocuments[0], classes, train_size=trainSize, stratify=classes
)
trainTweets, validationTweets, trainLabels, validationLabels = train_test_split(
  trainTweets, trainLabels, train_size=trainSize, stratify=trainLabels
)

tokenizer = Tokenizer(num_words=noOfWords, oov_token='<UNK>')

tokenizer.fit_on_texts(trainTweets)
trainPaddedSequences = GetPaddedSequences(tokenizer, trainTweets, maxLength=maxLength)

uniqueClasses = set(classes)
noOfClasses = len(uniqueClasses)
class2index = dict((c, i) for i, c in enumerate(uniqueClasses))
index2class = dict((v, k) for k, v in class2index.items())
names2ids = lambda labels: np.array([class2index.get(x) for x in trainLabels])
trainLabelsIDs = names2ids(trainLabels)
trainLabelsCat = to_categorical(trainLabelsIDs, num_classes=noOfClasses)

validationPaddedSequences = GetPaddedSequences(tokenizer, validationTweets, maxLength=maxLength)
names2ids = lambda labels: np.array([class2index.get(x) for x in validationLabels])
validationLabelsIDs = names2ids(validationLabels)
validationLabelsCat = to_categorical(validationLabelsIDs, num_classes=noOfClasses)

testPaddedSequences = GetPaddedSequences(tokenizer, testTweets, maxLength=maxLength)
names2ids = lambda labels: np.array([class2index.get(x) for x in testLabels])
testLabelsIDs = names2ids(testLabels)
testLabelsCat = to_categorical(testLabelsIDs, num_classes=noOfClasses)

print("Class2Index:", class2index)
print("Index2Class:", index2class)
print(f"The classes are: {uniqueClasses}.")

print(f"The first train padded sequence: ({trainPaddedSequences[0]}).")
print(f"The first validation padded sequence: ({validationPaddedSequences[0]}).")
print(f"The first test padded sequence: ({testPaddedSequences[0]}).")

model, history = ModelBiLSTM1()
PlotsHelper().PlotTrainingHistory(history)
model.load_weights("model.h5")
result = model.evaluate(testPaddedSequences, testLabels)
i = random.randint(0, len(testLabels) - 1)
p = model.predict(np.expand_dims(testPaddedSequences[i], axis=0))[0]
predClass = index2class[np.argmax(p).astype('uint8')]
testPred = np.argmax(model.predict(testPaddedSequences), axis=-1)
print('Sentence:', testTweets[i])
print('Category:', index2class[testLabelsIDs[i]])
print('Predicted Category:', predClass)
