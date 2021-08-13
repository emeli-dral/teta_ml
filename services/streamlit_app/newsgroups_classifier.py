import joblib
from sklearn.datasets import fetch_20newsgroups

class NewsgroupsClassifier(object):
	def __init__(self):
		self.vectorizer = joblib.load('text_vectorizer.pkl')
		self.model = joblib.load('text_classification_model.pkl')
		self.target_names = fetch_20newsgroups(subset='test').target_names

	def predict_text(self, text):
		try:
			vectorized_text = self.vectorizer.transform([text])
			return self.model.predict(vectorized_text)[0]
		except:
			print('Failed to predict')
			return None

	def get_name_by_label(self, label):
		try:
			return self.target_names[label]
		except:
			return 'label error'

	def get_topic(self, text):
		label = self.predict_text(text)
		return self.get_name_by_label(label)

