# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 18:36:35 2022

@author: abc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import pickle
import tensorflow as tf
import wordcloud

from PIL import Image
from pytesseract import pytesseract
import os
from os import listdir
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# helps in text preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

    # helps in model building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping

# split data into train and test set
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("E:/Dessertation/Dataset_5971/combine_spam_ham.csv",encoding= 'unicode_escape')


df.head()
data= df
data['LABEL'] = data['LABEL'].map({'spam': 0, 'ham': 1})
df.LABEL.value_counts()
df.shape

data_ham  = data[data['LABEL'] == 1].copy()
data_spam = data[data['LABEL'] == 0].copy()

def show_wordcloud(df, title):
    text = ' '.join(df['TEXT'].astype(str).tolist())
    stopwords = set(wordcloud.STOPWORDS)
    fig_wordcloud = wordcloud.WordCloud(stopwords=stopwords,background_color='lightgrey',
                    colormap='viridis', width=800, height=600).generate(text)
    plt.figure(figsize=(10,7), frameon=True)
    plt.imshow(fig_wordcloud)  
    plt.axis('off')
    plt.title(title, fontsize=20 )
    plt.show()

show_wordcloud(data_spam, "Spam messages")
show_wordcloud(data_ham, "Ham messages")


import collections
def word_count_plot(data):
     # finding words along with count
     word_counter = collections.Counter([word for sentence in data for word in sentence.split()])
     most_count = word_counter.most_common(30) # 30 most common words
     # sorted data frame
     most_count = pd.DataFrame(most_count, columns=["Word", "Count"]).sort_values(by="Count")
     most_count.plot.barh(x = "Word", y = "Count", color="green", figsize=(10, 15))
     
word_count_plot(df["TEXT"])



def preprocessing_cleantext(text):
    text = text.lower()
    text=text.replace("</p>","") # removing </p>
    text=text.replace("<p>"," ")  # removing <p>
    text = text.replace("http", " ")
    text = text.replace("www", " ")
    text = re.sub(r'([a-z])\1+', r'\1', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub('\.+', '.', text)
    text = re.sub(r"(?:\@|'|https?\://)\s+","",text) #delete punctuation
    text = re.sub("[^a-zA-Z]", " ",text)
    text=re.sub(r'[^\w\s]','',text) # remove punctuation
    text=re.sub("\d+","",text) # remove number from text
    tokens_text = nltk.word_tokenize(text) # tokenizing the documents
    stopwords=nltk.corpus.stopwords.words('english') #stopword reduction
    tokens_text=[w for w in tokens_text if w.lower() not in stopwords]
    tokens_text=[w.lower() for w in tokens_text] #convert to lower case
    tokens_text=[w for w in tokens_text if len(w)>2] #considering tokens with length>2(meaningful words)
    p= PorterStemmer() # stemming tokenized documents using Porter Stemmer
    tokens_text = [p.stem(w) for w in tokens_text]
    return text


df["TEXT"] = df["TEXT"].apply(preprocessing_cleantext) #clean_text
df["TEXT"]

X = data['TEXT'].values
y = data['LABEL'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# prepare tokenizer
t = Tokenizer()
X=t.fit_on_texts(X_train)   


# integer encode the documents
encoded_train = t.texts_to_sequences(X_train)
encoded_test = t.texts_to_sequences(X_test)
print(encoded_train[0:2])

max_length = 8
padded_train = pad_sequences(encoded_train, maxlen=max_length, padding='pre')
padded_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')
print(padded_train)

vocab_size = len(t.word_index) + 1

def c_report(y_true, y_pred):
   print("Classification Report")
   print(classification_report(y_true, y_pred))
   acc_sc = accuracy_score(y_true, y_pred)
   print("Accuracy : "+ str(acc_sc))
   return acc_sc

def plot_confusion_matrix(y_true, y_pred):
   mtx = confusion_matrix(y_true, y_pred)
   sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5, 
               cmap="Blues", cbar=False)
   plt.ylabel('True label')
   plt.xlabel('Predicted label')


############################ Flatten model creation #############################
# define the model
def create_flatten_model():
    model = Sequential()
    model.add(Embedding(vocab_size, 24, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

model= create_flatten_model()
# compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# fit the model
flatten_model= model.fit(x=padded_train,
         y=y_train,
         epochs=50,
         validation_data=(padded_test, y_test), verbose=1,
         callbacks=[early_stop]
         )

   
preds = (model.predict(padded_test) > 0.5).astype("int32")
c_report(y_test, preds)
plot_confusion_matrix(y_test, preds)

plt.plot(flatten_model.history['accuracy'])
plt.plot(flatten_model.history['val_accuracy'])
plt.title('RNN Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('Number of epochs')
plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='best')
plt.show()

plt.plot(flatten_model.history['loss'])
plt.plot(flatten_model.history['val_loss'])
plt.title('RNN Model loss')
plt.ylabel('loss')
plt.xlabel('Number of epochs')
plt.legend(['Training loss', 'Testing loss'], loc='best')
plt.show()


####################### flatten Model Creation ##############################

model.save("E:/Dessertation/Dataset_5971/fruad_model")

with open('E:/Dessertation/Dataset_5971/fruad_model/tokenizer.pkl', 'wb') as output:
    pickle.dump(t, output, pickle.HIGHEST_PROTOCOL)

s_model = tf.keras.models.load_model("E:/Dessertation/Dataset_5971/fruad_model")
with open('E:/Dessertation/Dataset_5971/fruad_model/tokenizer.pkl', 'rb') as input:
    tokenizer = pickle.load(input)
    
######################### LSTM part########################
from keras.layers import Dense, Embedding, LSTM, Dropout
def create_model():
      lstm_model = Sequential()
      lstm_model.add(Embedding(vocab_size, 24, input_length=max_length))
      lstm_model.add(LSTM(100))
      lstm_model.add(Dense(500, activation='relu'))
      lstm_model.add(Dense(200, activation='relu'))
      lstm_model.add(Dropout(0.5))
      lstm_model.add(Dense(100, activation='relu'))
      lstm_model.add(Dense(1, activation='sigmoid'))


      # lstm_model.add(Dropout(0.4))
      # lstm_model.add(Dense(20, activation="relu"))
      # lstm_model.add(Dropout(0.3))
      # lstm_model.add(Dense(1, activation = "sigmoid"))
      return lstm_model
lstm_model = create_model()
lstm_model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
print(lstm_model.summary())

# fit the model
lstm_model1=lstm_model.fit(x=padded_train,
         y=y_train,
         epochs=50,
         validation_data=(padded_test, y_test), verbose=1,
         callbacks=[early_stop]
         )

lstm_preds = (lstm_model.predict(padded_test) > 0.5).astype("int32")

c_report(y_test, lstm_preds)
plot_confusion_matrix(y_test, lstm_preds)


plt.plot(lstm_model1.history['accuracy'])
plt.plot(lstm_model1.history['val_accuracy'])
plt.title('LSTM Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('Number of epochs')
plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='best')
plt.show()

plt.plot(lstm_model1.history['loss'])
plt.plot(lstm_model1.history['val_loss'])
plt.title('LSTM Model loss')
plt.ylabel('loss')
plt.xlabel('Number of epochs')
plt.legend(['Training loss', 'Testing loss'], loc='best')
plt.show()

#############################END########################################   


####################### LSTM model creation #######################
lstm_model.save("E:/Dessertation/Dataset_5971/lstm_fruad_model")

with open('E:/Dessertation/Dataset_5971/lstm_fruad_model/lstm_tokenizer.pkl', 'wb') as output:
   pickle.dump(t, output, pickle.HIGHEST_PROTOCOL)

s_model_lstm = tf.keras.models.load_model("E:/Dessertation/Dataset_5971/lstm_fruad_model")
with open('E:/Dessertation/Dataset_5971/lstm_fruad_model/lstm_tokenizer.pkl', 'rb') as input:
    lstm_tokenizer = pickle.load(input)    
    
  
######################### Make Predictions with actual data ########################## 
    
def main():
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    pytesseract.tesseract_cmd = path_to_tesseract

files = os.listdir(r'E:\Dessertation\Final_Images - Copy')
sms_text= []

i = 0
while i in range(len(files)):
    #print(files[i])
        img = Image.open(os.path.join(r'E:\Dessertation\Final_Images - Copy', files[i]))
        text = pytesseract.image_to_string(img,lang="ENG")
        sms_text.append(text)
        i=i+1
         
if __name__ == '__main__':
    main() 
    
sms_text_data= {'index':[],'value':[]} 
for index, value in enumerate(sms_text):
    sms_text_data['index'].append(index)
    sms_text_data['value'].append((value))
    
sms_text_data_df = pd.DataFrame(sms_text_data)	

sms_text_data_df["value"] = sms_text_data_df["value"].apply(preprocessing_cleantext) #clean_text
sms_text_data_df["value"]

import numpy as np
sms_text_data_df=sms_text_data_df[sms_text_data_df['value'].str.strip().astype(bool)]

	 

################ Flatten prediction#########
c=['sms_text','pred']
sms_fraud_detection= pd.DataFrame(columns=c)

for i in sms_text_data_df:
    word=sms_text_data_df['value']
    sms_proc = tokenizer.texts_to_sequences(word.values)
    sms_proc = pad_sequences(sms_proc, maxlen=max_length, padding='post')
    pred = (model.predict(sms_proc) > 0.5).astype("int32")
    sms_fraud_detection['sms_text']=word.values
    sms_fraud_detection['pred']=pred.tolist()

sms_fraud_detection.pred.value_counts() # 0 spam  1 ham


#################lstm prediction########

lstm_c=['sms_text','pred']
lstm_sms_fraud_detection= pd.DataFrame(columns=c)

for i in sms_text_data_df:
    word=sms_text_data_df['value']
    sms_proc = lstm_tokenizer.texts_to_sequences(word.values)
    sms_proc = pad_sequences(sms_proc, maxlen=max_length, padding='post')
    pred = (lstm_model.predict(sms_proc) > 0.5).astype("int32")
    lstm_sms_fraud_detection['sms_text']=word.values
    lstm_sms_fraud_detection['pred']=pred.tolist()

lstm_sms_fraud_detection.pred.value_counts()

######################### BI-LSTM part########################
from keras.layers import Bidirectional
def create_model_bilstm():
      bi_lstm_model = Sequential()
      bi_lstm_model.add(Embedding(vocab_size, 24, input_length=max_length))
      bi_lstm_model.add(Bidirectional(LSTM(100)))
      bi_lstm_model.add(Dense(500, activation='relu'))
      bi_lstm_model.add(Dense(200, activation='relu'))
      bi_lstm_model.add(Dropout(0.5))
      bi_lstm_model.add(Dense(100, activation='relu'))
      bi_lstm_model.add(Dense(1, activation='sigmoid'))
      
      
      
      # bi_lstm_model.add(Dropout(0.4))
      # bi_lstm_model.add(Dense(20, activation="relu"))
      # bi_lstm_model.add(Dropout(0.3))
      # bi_lstm_model.add(Dense(1, activation = "sigmoid"))
      return bi_lstm_model
bi_lstm_model = create_model_bilstm()
bi_lstm_model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
print(bi_lstm_model.summary())

# fit the model
bi_lstm_model1=bi_lstm_model.fit(x=padded_train,
         y=y_train,
         epochs=50,
         validation_data=(padded_test, y_test), verbose=1,
         callbacks=[early_stop]
         )

bi_lstm_preds = (bi_lstm_model.predict(padded_test) > 0.5).astype("int32")

c_report(y_test, bi_lstm_preds)
plot_confusion_matrix(y_test, bi_lstm_preds)

plt.plot(bi_lstm_model1.history['accuracy'])
plt.plot(bi_lstm_model1.history['val_accuracy'])
plt.title('Bi-LSTM Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('Number of epochs')
plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='best')
plt.show()

plt.plot(bi_lstm_model1.history['loss'])
plt.plot(bi_lstm_model1.history['val_loss'])
plt.title('Bi-LSTM Model loss')
plt.ylabel('loss')
plt.xlabel('Number of epochs')
plt.legend(['Training loss', 'Testing loss'], loc='best')
plt.show()



#############################END########################################  

####################### Bi-LSTM model creation #######################
bi_lstm_model.save("E:/Dessertation/Dataset_5971/bi_lstm_fruad_model")

with open('E:/Dessertation/Dataset_5971/bi_lstm_fruad_model/bi_lstm_tokenizer.pkl', 'wb') as output:
   pickle.dump(t, output, pickle.HIGHEST_PROTOCOL)

s_model_bi_lstm = tf.keras.models.load_model("E:/Dessertation/Dataset_5971/bi_lstm_fruad_model")
with open('E:/Dessertation/Dataset_5971/bi_lstm_fruad_model/bi_lstm_tokenizer.pkl', 'rb') as input:
    bi_lstm_tokenizer = pickle.load(input)    
    
    
#################bi_lstm prediction########

bi_lstm_c=['sms_text','pred']
bi_lstm_sms_fraud_detection= pd.DataFrame(columns=c)

for i in sms_text_data_df:
    word=sms_text_data_df['value']
    sms_proc = bi_lstm_tokenizer.texts_to_sequences(word.values)
    sms_proc = pad_sequences(sms_proc, maxlen=max_length, padding='post')
    pred = (bi_lstm_model.predict(sms_proc) > 0.5).astype("int32")
    bi_lstm_sms_fraud_detection['sms_text']=word.values
    bi_lstm_sms_fraud_detection['pred']=pred.tolist()

bi_lstm_sms_fraud_detection.pred.value_counts()





