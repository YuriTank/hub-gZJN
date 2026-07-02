import torch
import torch.nn as nn
import torch.nn.functional as F

class TweetClassifier(nn.Module):

	def __init__(self, args):
		super(TweetClassifier, self).__init__()
		
		self.hidden_dim = args.hidden_dim
		self.LSTM_layers = args.lstm_layers
		self.input_size = args.max_words
		self.bidirectional = True
		
		self.embedding_dropout = nn.Dropout(0.3)
		self.dropout = nn.Dropout(0.7)
		self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
		self.lstm = nn.LSTM(
			input_size=self.hidden_dim,
			hidden_size=self.hidden_dim,
			num_layers=self.LSTM_layers,
			batch_first=True,
			bidirectional=self.bidirectional,
		)
		lstm_output_dim = self.hidden_dim * 2  # bidirectional
		self.fc1 = nn.Linear(lstm_output_dim, self.hidden_dim)
		self.fc2 = nn.Linear(self.hidden_dim, 1)
		
	def forward(self, x):
		device = x.device
		batch_size = x.size(0)
		
		num_directions = 2  # bidirectional
		h = torch.zeros(self.LSTM_layers * num_directions, batch_size, self.hidden_dim, device=device)
		c = torch.zeros(self.LSTM_layers * num_directions, batch_size, self.hidden_dim, device=device)

		out = self.embedding(x)
		out = self.embedding_dropout(out)
		out, _ = self.lstm(out, (h, c))
		out = out[:, -1, :]  # last timestep output
		out = self.dropout(out)
		out = F.relu(self.fc1(out))
		out = self.dropout(out)
		out = torch.sigmoid(self.fc2(out))

		return out
