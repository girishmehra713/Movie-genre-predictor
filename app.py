from flask import Flask ,render_template,request
import joblib
import numpy as np
import nltk
sw = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

ps = PorterStemmer()
tokenizer = RegexpTokenizer('[a-z]+')

def clean_review(review):
      review = review.lower()
      tokens = tokenizer.tokenize(review)
      tokens = [token for token in tokens if not token in sw]
      cleanedReview = [ps.stem(token) for token in tokens]
      cleanedReview = ' '.join(cleanedReview)
      return [cleanedReview]

cv = joblib.load("transform.pkl")
mnb = joblib.load("model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
	return render_template("design.html")

@app.route('/predict',methods = ['POST'])
def result():
	if request.method == 'POST':
		dialog = request.form['review']
		dialog = clean_review(dialog)
		vect = cv.transform(dialog)
		my_prediction = mnb.predict(vect)
		return render_template("result.html",prediction = my_prediction)

if __name__ == '__main__':
		app.run(debug = True)