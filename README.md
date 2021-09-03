# Spam Detector Flask App

<b> Hosted at: </b> https://spam-detector-300821.herokuapp.com/

The Spam Detector is a Flask web application which predicts whether the message is <b> Spam </b> or <b> Ham</b>.

The dataset is taken from Kaggle and is used for classification of the message. The text is cleaned by expanding contractions, converting the text to lower case, removing digits and punctuations and applying lemmatization and removing stopwords. After cleaning, the 'Category' attribute is encoded where 0->Ham an 1->Spam. Then, CountVectorizer() is applied on 'Message' attribute to convert text into vectors.
