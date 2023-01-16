
#Libraries for web application and image processing
from flask import Flask, render_template, request,flash,url_for,redirect
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
from PIL import Image
from pytesseract import pytesseract
import os
from os import listdir

#Libraries to load stored model and preprocessing of K-means and RNN
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#Loading model through pickle
kmodel = pickle.load(open('kmodel.pkl', 'rb'))
ktfidf = pickle.load(open('ktfidf.pkl', 'rb'))

KNNmodel = tf.keras.models.load_model("C:/Users/abc/Spam_Detection/Flatten/fruad_model")
KNNtfidf = pickle.load(open('C:/Users/abc/Spam_Detection/Flatten/fruad_model/tokenizer.pkl', 'rb'))

classifier = kmodel
cv = ktfidf

KNNclassifier = KNNmodel
KNNcv = KNNtfidf

app = Flask(__name__)
app.config['SECRET_KEY'] = '5b703e19aefc754d7cc17362a6c8387bdf75a4a2254a51e9'
app.config["UPLOAD_FOLDER"] = "E:/Dessertation/Dataset_5971/Flask-file-upload-main/Flask-file-upload-main"


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

@app.route('/',methods = ['GET', 'POST'])
def upload_file():
    return render_template('index.html')

@app.route('/message', methods = ['GET', 'POST'])
def message():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            flash('No selected file', 'danger')
            return render_template('index.html')
        
                
@app.route('/displaymodels', methods = ['GET', 'POST'])
def model_output():
    model = ['kmeans', 'RNN']
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(app.config['UPLOAD_FOLDER'] + filename)
        path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        path_to_image = r'E:/Dessertation/Dataset_5971/Flask-file-upload-main/Flask-file-upload-main'+filename
        img = Image.open(path_to_image)
        sms_text = [pytesseract.image_to_string(img)]
        content=sms_text
        sms_df = pd.DataFrame(sms_text, columns=['message'])
        content=sms_df["message"].apply(preprocessing_cleantext)
        content_df = pd.DataFrame(content) 
        if request.form.get("kmeans"):
            sms_proc = ktfidf.transform(content_df["message"]).toarray()
            my_prediction = (classifier.predict(sms_proc))
            model="kmeans"
        elif request.form.get("RNN"): 
            max_length = 8
            sms_proc = KNNtfidf.texts_to_sequences(content)
            sms_proc = pad_sequences(sms_proc, maxlen=max_length, padding='post')
            my_prediction = (KNNclassifier.predict(sms_proc) > 0.5).astype("int32")
            model="RNN"
    return render_template('content_kmeans-KNN.html', prediction=my_prediction,model=model)


@app.route('/displayallmodels', methods = ['GET', 'POST'])
def allmodel_output():
    KGmodel = ['kmeans', 'RNN']
    if request.method == 'POST':
       f = request.files['file']
       filename = secure_filename(f.filename)
       f.save(app.config['UPLOAD_FOLDER'] + filename)
       path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
       path_to_image = r'E:/Dessertation/Dataset_5971/Flask-file-upload-main/Flask-file-upload-main'+filename
       img = Image.open(path_to_image)
       sms_text = [pytesseract.image_to_string(img)]
       content=sms_text
       sms_df = pd.DataFrame(sms_text, columns=['message'])
       content=sms_df["message"].apply(preprocessing_cleantext)
       content_df = pd.DataFrame(content) 
       if request.form.get("kmeans") and request.form.get("RNN"):
            Ksms_proc = ktfidf.transform(content_df["message"]).toarray()
            kmy_prediction = (classifier.predict(Ksms_proc))
            max_length = 8
            KNNsms_proc = KNNtfidf.texts_to_sequences(content)
            KNNsms_proc = pad_sequences(KNNsms_proc, maxlen=max_length, padding='post')
            KNNmy_prediction = (KNNclassifier.predict(KNNsms_proc) > 0.5).astype("int32")
            if kmy_prediction == KNNmy_prediction:
                   my_prediction=kmy_prediction
            elif kmy_prediction !=  KNNmy_prediction:
                   my_prediction=''
                   kmy_prediction = kmy_prediction
                   KNNmy_prediction = KNNmy_prediction
    return render_template('content_allmodels-KM-KN.html', prediction=my_prediction,kpred=kmy_prediction,Knpred=KNNmy_prediction)



if __name__ == '__main__':
    app.run()



