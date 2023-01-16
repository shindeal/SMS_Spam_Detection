
#Import libraries and packages
from PIL import Image
from pytesseract import pytesseract
import os
from os import listdir
import pandas as pd
from sklearn.cluster import KMeans
import csv
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import re
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords

# Lets do some text cleanup
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
REMOVE_NUM = re.compile('[\d+]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = text.replace('x', '') # Remove the XXXX values
    text = REMOVE_NUM.sub('', text)# Remove white space
    text = BAD_SYMBOLS_RE.sub('', text) #  delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwords from text
    text = ' '.join(word for word in text.split() if (len(word) > 2 )) #and len(word) <= 21 # removes any words composed of less than 2 or more than 21 letters
    text = ' '.join([stemmer.stem(word) for word in text.split()]) # Stemming the words
    return text

def preprocessing_cleantext(text):
    text = text.lower()
    text=text.replace("</p>","") # removing </p>
    text=text.replace("<p>"," ")  # removing <p>
    text = text.replace("http", " ")
    text = text.replace("www", " ")
    text = re.sub(r'([a-z])\1+', r'\1', text)
    text = re.sub('\s+', ' ', text) #white space
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

sms_text_data_df = pd.read_csv("E:\Dessertation\Python_code\Final_code\spam_ham_dataset.csv",encoding= 'unicode_escape')
sms_text_data_df.value.value_counts()


sms_text_data_df["value"] = sms_text_data_df["value"].apply(preprocessing_cleantext) #,preprocessing_cleantext
sms_text_data_df["value"]

import numpy as np
sms_text_data_df['value'].replace('', np.nan, inplace=True)
sms_text_data_df.dropna(subset=['value'], inplace=True)
sms_text_data_df=sms_text_data_df[sms_text_data_df['value'].str.strip().astype(bool)]


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=10)
X_train_vc = vectorizer.fit_transform(sms_text_data_df["value"])
pd.DataFrame(X_train_vc.toarray(), columns=vectorizer.get_feature_names()).head()
print(X_train_vc.shape)
print(vectorizer.vocabulary_)

k_clusters = 2
model = KMeans(n_clusters=k_clusters, init='k-means++', n_init=10, max_iter=100, tol=0.000001,copy_x=True,random_state=99 )
model.fit(X_train_vc)
clusters = model.predict(X_train_vc)
centers = model.cluster_centers_
labels = model.labels_

sms_text_data_df["ClusterName"] = clusters
sms_text_data_df.head(20)

list_true = []
list_true=sms_text_data_df['Label']

list_prediction=[]
list_prediction=sms_text_data_df['ClusterName']

accuracy =  accuracy_score(list_true, list_prediction)
print ("Accuracy score:", accuracy)

sms_text_data_df["ClusterName"].value_counts()



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
   plt.title('Kmeans Confusion Matrix')


c_report(list_true, list_prediction)
plot_confusion_matrix(list_true, list_prediction)

import pickle
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('tfidf.pkl', 'wb'))
model_pre_trained = pickle.load(open('model.pkl','rb'))


model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))


#####################new data for prediction #################

path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.tesseract_cmd = path_to_tesseract

files = os.listdir(r'E:\Dessertation\Final_Images - Copy')
sms_text_new_data_pred= []

i = 0
while i in range(len(files)):
    #print(files[i])
        img = Image.open(os.path.join(r'E:\Dessertation\Final_Images - Copy', files[i]))
        text = pytesseract.image_to_string(img,lang="ENG")
        sms_text_new_data_pred.append(text)
        i=i+1

sms_text_data_pred= {'index':[],'value':[]} 
for index, value in enumerate(sms_text_new_data_pred):
    sms_text_data_pred['index'].append(index)
    sms_text_data_pred['value'].append((value))
    
sms_text_data_pred_df = pd.DataFrame(sms_text_data_pred)

sms_text_data_pred_df["value"] = sms_text_data_pred_df["value"].apply(preprocessing_cleantext) #clean_text
sms_text_data_pred_df["value"]

import numpy as np
sms_text_data_pred_df=sms_text_data_pred_df[sms_text_data_pred_df['value'].str.strip().astype(bool)]


model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

sms_text_unseen = tfidf.transform(sms_text_data_pred_df["value"]).toarray()
print(sms_text_unseen.shape)
cluster_pred_unseen = model_pre_trained.predict(sms_text_unseen)
print(cluster_pred_unseen)

sms_text_data_pred_df["ClusterName"] = cluster_pred_unseen
sms_text_data_pred_df.head(20)
sms_text_data_pred_df["ClusterName"].value_counts()

####################################

