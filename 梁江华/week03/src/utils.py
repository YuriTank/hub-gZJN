import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split

# Local Tokenizer implementation
class SimpleTokenizer:
	def __init__(self, num_words=None):
		self.num_words = num_words
		self.word_index = {}
		self.index_word = {}
		self.fitted = False

	def fit_on_texts(self, texts):
		word_freq = {}
		for text in texts:
			for word in str(text).split():
				word_freq[word] = word_freq.get(word, 0) + 1
		# Sort by frequency
		sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
		if self.num_words:
			sorted_words = sorted_words[:self.num_words-1]
		self.word_index = {w: i+1 for i, (w, _) in enumerate(sorted_words)}
		self.index_word = {i: w for w, i in self.word_index.items()}
		self.fitted = True

	def texts_to_sequences(self, texts):
		if not self.fitted:
			raise ValueError("Tokenizer has not been fitted yet.")
		sequences = []
		for text in texts:
			seq = [self.word_index.get(word, 0) for word in str(text).split()]
			sequences.append(seq)
		return sequences

# Local pad_sequences implementation
def pad_sequences(sequences, maxlen):
	padded = np.zeros((len(sequences), maxlen), dtype=int)
	for idx, seq in enumerate(sequences):
		if not seq:
			continue
		if len(seq) > maxlen:
			padded[idx] = seq[-maxlen:]
		else:
			padded[idx, -len(seq):] = seq
	return padded


class Preprocessing:
	
	def __init__(self, args):
		self.y_test = None
		self.y_train = None
		self.x_test = None
		self.x_train = None
		self.tokens = None
		default_data_path = Path(__file__).resolve().parent.parent / 'data' / 'tweets.csv'
		data_path = getattr(args, 'data_path', None) or default_data_path
		self.data = Path(data_path).resolve()
		self.max_len = args.max_len
		self.max_words = int(args.max_words)
		self.test_size = args.test_size
		
	def load_data(self):
		if not self.data.exists():
			raise FileNotFoundError(f"Dataset file not found: {self.data}")
		df = pd.read_csv(self.data)
		df.drop(['id','keyword','location'], axis=1, inplace=True)
		
		x = df['text'].values
		y = df['target'].values
		
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=self.test_size)
		
	def prepare_tokens(self):
		self.tokens = SimpleTokenizer(num_words=self.max_words)
		self.tokens.fit_on_texts(self.x_train)

	def sequence_to_token(self, x):
		sequences = self.tokens.texts_to_sequences(x)
		return pad_sequences(sequences, maxlen=self.max_len)
