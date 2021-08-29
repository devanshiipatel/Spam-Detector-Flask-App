from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle, re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Convenient way to get the import name of the place the app is defined
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('spam_text_message.csv')
    
    # Text Preprocessing
    
    # Expanding contractions

    # Dictionary of English Contractions
    contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                        "can't": "cannot","can't've": "cannot have",
                        "'cause": "because","could've": "could have","couldn't": "could not",
                        "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                        "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                        "hasn't": "has not","haven't": "have not","he'd": "he would",
                        "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                        "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                        "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                        "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                        "it'd": "it would","it'd've": "it would have","it'll": "it will",
                        "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                        "mayn't": "may not","might've": "might have","mightn't": "might not", 
                        "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                        "mustn't've": "must not have", "needn't": "need not",
                        "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                        "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                        "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                        "she'll": "she will", "she'll've": "she will have","should've": "should have",
                        "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                        "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                        "there'd've": "there would have", "they'd": "they would",
                        "they'd've": "they would have","they'll": "they will",
                        "they'll've": "they will have", "they're": "they are","they've": "they have",
                        "to've": "to have","wasn't": "was not","we'd": "we would",
                        "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                        "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                        "what'll've": "what will have","what're": "what are", "what've": "what have",
                        "when've": "when have","where'd": "where did", "where've": "where have",
                        "who'll": "who will","who'll've": "who will have","who've": "who have",
                        "why've": "why have","will've": "will have","won't": "will not",
                        "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                        "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                        "y'all'd've": "you all would have","y'all're": "you all are",
                        "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                        "you'll": "you will","you'll've": "you will have", "you're": "you are",
                        "you've": "you have"}
    # Regular expression for finding contractions
    contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

    # Function for expanding contractions
    def expand_contractions(text,contractions_dict=contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, text)

    # Expanding Contractions in the reviews
    # df['Message'] = df['Message'].apply(lambda x:expand_contractions(x))
    
    # Converting text to lowercase
    # df['Message'] = df['Message'].apply(lambda x:x.lower())
    
    # Removing digits and words containing digits
    # df['Message'] = df['Message'].apply(lambda x: re.sub('\w*\d\w*','', x))
    
    # Removing punctuations
    # df['Message'] = df['Message'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    
    # Removing extra spaces
    # df['Message']=df['Message'].apply(lambda x: re.sub(' +',' ',x))
    
    # Apply Lemmatization and Remove stopwords
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def lemmatize_text(text):
        wn.ensure_loaded()
        rev = [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text) if w not in stopwords.words('english')]
        rev = ' '.join(rev)
        return rev

    # df['Message'] = df.Message.apply(lemmatize_text)
    
    # Model
    
    # Creating a Bag of Words model
    cv = CountVectorizer()
    X = cv.fit_transform(df['Message']).toarray()
    
    # Encoding dependent variable
    le = LabelEncoder()
    df["Category"] = le.fit_transform(df["Category"])    # ham->0 and spam->1
    
    y = df["Category"]
    
    # Split the dataset into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Import the model
    NB_spam_model = open('NB_spam_model.pkl','rb')
    clf = joblib.load(NB_spam_model)
    clf.fit(X_train, y_train)
    score = clf.score(X_test,y_test)
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True, port=9999)
 
