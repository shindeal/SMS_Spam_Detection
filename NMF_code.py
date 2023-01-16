# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 12:43:43 2022

@author: abc
"""

from PIL import Image
from pytesseract import pytesseract
import os
from os import listdir
import pandas as pd
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import re
import nltk
from nltk.corpus import stopwords

from sklearn.metrics import accuracy_score

# def main():
#     path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#     pytesseract.tesseract_cmd = path_to_tesseract

# files = os.listdir(r'E:\Dessertation\Final_Images')
# sms_text= []

# i = 0
# while i in range(len(files)):
#     #print(files[i])
#         img = Image.open(os.path.join(r'E:\Dessertation\Final_Images', files[i]))
#         text = pytesseract.image_to_string(img,lang="ENG")
#         sms_text.append(text)
#         i=i+1
         
# if __name__ == '__main__':
#     main() 
    
# nmf= {'index':[],'value':[]} 
# for index, value in enumerate(sms_text):
#     nmf['index'].append(index)
#     nmf['value'].append((value))
    
# nmf_df = pd.DataFrame(nmf)	


#nmf_df = pd.read_csv("E:\Dessertation\Python_code\combine_spam_ham.csv",encoding= 'unicode_escape')
#nmf_df.LABEL.value_counts()

nmf_df = pd.read_csv("E:\Dessertation\Python_code\Final_code\spam_ham_dataset.csv",encoding= 'unicode_escape')
nmf_df.value.value_counts()

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

nmf_df = pd.read_csv("E:\Dessertation\Python_code\Final_code\spam_ham_dataset.csv",encoding= 'unicode_escape')
nmf_df.value.value_counts()

nmf_df["value"] = nmf_df["value"].apply(preprocessing_cleantext) #clean_text
nmf_df["value"]

import numpy as np
nmf_df['value'].replace('', np.nan, inplace=True)
nmf_df.dropna(subset=['value'], inplace=True)
nmf_df=nmf_df[nmf_df['value'].str.strip().astype(bool)]

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_nmf = TfidfVectorizer(min_df=10)
X_train_nmf = tfidf_nmf.fit_transform(nmf_df["value"])
pd.DataFrame(X_train_nmf.toarray(), columns=tfidf_nmf.get_feature_names()).head()
print(X_train_nmf.shape)






from sklearn.decomposition import NMF
nmf_model = NMF(n_components=2,max_iter=600,solver="mu")
nmf_model.fit(X_train_nmf)



#print("record size : ",len(tfidf_nmf.get_feature_names()))

# import random
# for i in range(10):
#     random_word_id = random.randint(0,len(tfidf_nmf.get_feature_names()))
#     print(tfidf_nmf.get_feature_names()[random_word_id])
    
# for index,topic in enumerate(nmf_model.components_):
#     print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
#     print([tfidf_nmf.get_feature_names()[i] for i in topic.argsort()[-15:]])
#     print('\n')
    
    
nmf_results = nmf_model.transform(X_train_nmf)
nmf_df['NMF_label'] = nmf_results.argmax(axis=1)
nmf_df.head(10)
print(nmf_df.NMF_label.value_counts())


list_true_nmf = []
list_true_nmf=nmf_df['Label']

list_prediction_nmf=[]
list_prediction_nmf=nmf_df['NMF_label']

accuracy =  accuracy_score(list_true_nmf, list_prediction_nmf)
print ("Accuracy score:", accuracy)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def c_report(y_true, y_pred):
   print("Classification Report")
   print(classification_report(y_true, y_pred))
   acc_sc = accuracy_score(y_true, y_pred)
   print("Accuracy : "+ str(acc_sc))
   return acc_sc


import matplotlib.pyplot as plt
import seaborn as sns
def plot_confusion_matrix(y_true, y_pred):
   mtx = confusion_matrix(y_true, y_pred)
   sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5, 
               cmap="Blues", cbar=False)
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.title('NMF Confusion Matrix')


c_report(list_true_nmf, list_prediction_nmf)
plot_confusion_matrix(list_true_nmf, list_prediction_nmf)

import pickle
pickle.dump(nmf_model, open('model.pkl', 'wb'))
pickle.dump(tfidf_nmf, open('tfidf.pkl', 'wb'))
model_pre_trained_nmf = pickle.load(open('model.pkl','rb'))

model_nmf = pickle.load(open('model.pkl', 'rb'))
tfidf_nmf = pickle.load(open('tfidf.pkl', 'rb'))

# print(nmf_results.shape)
# print(nmf_model.components_.shape)

# components_df = pd.DataFrame(nmf_model.components_, columns=tfidf_nmf.get_feature_names())
# components_df

# for topic in range(components_df.shape[0]):
#     tmp = components_df.iloc[topic]
#     print(f'For topic {topic+1} the words with the highest value are:')
#     print(tmp.nlargest(10))
#     print('\n')

###########New sms to predict####################

path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.tesseract_cmd = path_to_tesseract

files = os.listdir(r'E:\Dessertation\Final_Images')
sms_text_new_data_pred_nmf= []

i = 0
while i in range(len(files)):
    #print(files[i])
        img = Image.open(os.path.join(r'E:\Dessertation\Final_Images', files[i]))
        text = pytesseract.image_to_string(img,lang="ENG")
        sms_text_new_data_pred_nmf.append(text)
        i=i+1

sms_text_data_pred_nmf= {'index':[],'value':[]} 
for index, value in enumerate(sms_text_new_data_pred_nmf):
    sms_text_data_pred_nmf['index'].append(index)
    sms_text_data_pred_nmf['value'].append((value))
    
sms_text_data_pred_nmfdf = pd.DataFrame(sms_text_data_pred_nmf)

sms_text_data_pred_nmfdf["value"] = sms_text_data_pred_nmfdf["value"].apply(preprocessing_cleantext) #clean_text
sms_text_data_pred_nmfdf["value"]

import numpy as np
sms_text_data_pred_nmfdf=sms_text_data_pred_nmfdf[sms_text_data_pred_nmfdf['value'].str.strip().astype(bool)]

model_nmf = pickle.load(open('model.pkl', 'rb'))
tfidf_nmf = pickle.load(open('tfidf.pkl', 'rb'))

unseen_data = tfidf_nmf.transform(sms_text_data_pred_nmfdf["value"])
print(unseen_data.shape)
#unseen_cluster=nmf_model.transform(unseen_data)
unseen_cluster=model_nmf.transform(unseen_data)


pd.DataFrame(unseen_cluster)
pd.DataFrame(unseen_cluster).idxmax(axis=1)


sms_text_data_pred_nmfdf['label']=unseen_cluster.argmax(axis=1) #0 spam and 1 ham
sms_text_data_pred_nmfdf.label.value_counts()