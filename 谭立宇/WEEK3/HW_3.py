import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import string 
import numpy as np

COMMON_CHARS = "的一了是我不在人们有来他这上着个地到大里说就去子得也和那要下看天时过出小么起你都把好还多没为又可家学只以会到如长作些开三心但全对能明行最前头"

def generate_samples(n_sample, seed=42):
    # generate dataset: use 5 hanzi, "you" in the position 0-4, label is the index
    random.seed(seed) # set random seed
    samples = [] # initialize empty list for storing sample
    for i in range(n_sample):
        # use loop to generate certain amount of smaples
        pos = random.randint(0, 4) # "you" position, random generate for one of 5 position 
        chars = [random.choice(COMMON_CHARS) for j in range(5)]
        # randomly select 5 characters in COMMON CAHRS to forming the list
        chars[pos] = "你"
        # replace the characters in selected position to "you" 
        samples.append(("".join(chars), pos))
        # use the lists forming strings, and forming the tuples with position adding in samples lists
    return samples


# customize the dataset class, inherit from Pytorch's Dataset 
class CharDataset(Dataset):
    def __init__(self, samples):
        # initialize function, receive the sample list
        self.samples = samples
        # save sample list in instance variable
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        # initialize the vocabulary dictionary, including special indicator: PAD (panding) & UNK (unknown)
        idx = 2
        # use index 2 for real index assignment
        for text, i in samples:
            # interate all texts in sample
            for ch in text:
                # iterate characters in text
                if ch not in self.vocab:
                    # if character isn't in vocabulary, then add and reassign the new index
                    self.vocab[ch] = idx
                    # index adding
                    idx += 1

    # return amount of sample in dataset
    def __len__(self):
        return len(self.samples)
    
    # According to index, receive single sample, let the Dataloader use this function
    def __getitem__(self, idx):
        # receive text and label of certain index from sample list
        text, label = self.samples[idx]
        # tansfer each character in text to relative index of vocabulary, unknown character using UNK's index
        x = [self.vocab.get(ch, self.vocab["<UNK>"]) for ch in text]
        # transfer the index list and label as Pytorch long tensors and return
        return torch.tensor(x, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
"""
MODEL DEFINITION
"""
# define RNN model, inherit from Pytorch's Module class
class RNNClassifier(nn.Module):
    # initialize the function, parameters: vocabulary size, embedding dimension, hidden dimension, class amount, layer amount, dropout rate
    def __init__(self, vocab_size, embed_dim, hidden_dim, class_amount, num_layers=1, dropout=0.0):
        # use Module's initialize function
        super().__init__()
        # Create the embedding layer, convert character indexs into dense vectors.
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # create RNN layer, parameter: input dimension = embeded layer, hidden dimension, layer amount
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        # create a fully connected layer to map the RNN hidden states to the number of classes (5 classes).
        self.fc = nn.Linear(hidden_dim, class_amount)
    
    # define the forward function
    def forward(self, x):
        # use embedding layer to convert character vocabulary to embedding vectors
        emb = self.embedding(x)
        #  Pass embeddings through RNN, returns all time step outputs (_) and hidden state of the last time step
        # hidden shape: (num_layers, batch_size, hidden_dim)
        output, hidden = self.rnn(emb)
        # hidden: (num_layers, batch, hidden_dim) -> take the last layer's hidden state
        # Pass the last layer hidden state through the fully connected layer to get classification logits, shape: (batch_size, class_amount)
        out = self.fc(hidden[-1])
        # Return classification output
        return out
# Define LSTM classifier model, inheriting from PyTorch's Module base class
class LSTMClassifier(nn.Module):
    # Initialization method, parameters similar to RNNClassifier
    def __init__(self, vocab_size, embed_dim, hidden_dim, class_amount, num_layers=1, dropout=0.0):
        # Call parent Module's initialization method
        super().__init__()
        # Create embedding layer, converts character indices to dense vector representations
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Create LSTM layer, parameters similar to RNN, but LSTM has two states: cell state and hidden state
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        # Create fully connected layer, maps LSTM hidden state to num classes
        self.fc = nn.Linear(hidden_dim, class_amount)
    # Define forward propagation process
    def forward(self, x):
        # Convert character indices to embedding vectors via embedding layer
        emb = self.embedding(x)
        # Pass embeddings through LSTM, returns all time step outputs (_), hidden state of last time step, and cell state (_)
        # hidden shape: (num_layers, batch_size, hidden_dim)
        _, (hidden, _) = self.lstm(emb)
        # Pass the last layer hidden state through the fully connected layer to get classification logits
        out = self.fc(hidden[-1])
        # Return classification output
        return out
"""
3. Training and Evaluation Functions
"""
# Define function to train for one epoch, parameters: model, data loader, loss function, optimizer, device
def train_epoch(model, loader, criterion, optimizer, device):
    # Set model to training mode (enables dropout, etc.)
    model.train()
    # Initialize accumulated loss, correct predictions, total samples to 0
    total_loss, correct, total = 0, 0, 0
    # Iterate through each batch in the data loader
    for x, y in loader:
        # Move input data and labels to specified device (CPU or GPU)
        x, y = x.to(device), y.to(device)
        # Zero gradients in optimizer to prevent gradient accumulation
        optimizer.zero_grad()
        # Forward propagation: pass input through model to get logits (unnormalized prediction scores)
        logits = model(x)
        # Compute cross-entropy loss
        loss = criterion(logits, y)
        # Backward propagation: compute gradients of loss with respect to all parameters
        loss.backward()
        # Update model parameters
        optimizer.step()
        # Accumulate loss (loss.item() is the average loss of current batch, multiply by batch size to get total loss)
        total_loss += loss.item() * x.size(0)
        # Count correct predictions: argmax(1) returns the index of max value per row (i.e., predicted class), compare with true labels
        correct += (logits.argmax(1) == y).sum().item()
        # Accumulate sample count of current batch
        total += x.size(0)
    # Return average loss and accuracy
    return total_loss / total, correct / total
# Define evaluation function, parameters similar to train_epoch, but does not need optimizer
def evaluate(model, loader, criterion, device):
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()
    # Initialize accumulated loss, correct predictions, total samples to 0
    total_loss, correct, total = 0, 0, 0
    # Use torch.no_grad() context manager to prevent gradient computation, saving memory and computation
    with torch.no_grad():
        # Iterate through each batch in the data loader
        for x, y in loader:
            # Move input data and labels to specified device
            x, y = x.to(device), y.to(device)
            # Forward propagation to get logits
            logits = model(x)
            # Compute loss
            loss = criterion(logits, y)
            # Accumulate loss
            total_loss += loss.item() * x.size(0)
            # Count correct predictions
            correct += (logits.argmax(1) == y).sum().item()
            # Accumulate sample count
            total += x.size(0)
    # Return average loss and accuracy
    return total_loss / total, correct / total
"""
4. Main Pipeline
"""
# Define function to run a single model experiment, parameters: model name, model class, parameter dictionary
def run_experiment(model_name, model_cls, params):
    # Print separator line
    print(f"\n{'='*60}")
    # Print current experiment model name
    print(f"  Model: {model_name}")
    # Print separator line
    print(f"{'='*60}")
    # Detect if GPU is available, use "cuda" if yes, otherwise use "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate 10000 sample data points, use seed 42 for reproducibility
    samples = generate_samples(10000, seed=42)
    # Randomly shuffle all samples
    random.shuffle(samples)
    # Training set: first 8000 samples
    train_samples = samples[:8000]
    # Validation set: samples 8000 to 9000
    val_samples = samples[8000:9000]
    # Test set: samples 9000 to 10000
    test_samples = samples[9000:]
    # Create training dataset object, will build vocabulary
    train_ds = CharDataset(train_samples)
    # Reuse the same vocabulary (training set already contains all common characters)
    # Define internal test dataset class, uses training set's vocabulary to ensure index consistency
    class TestDataset(Dataset):
        # Initialization method, receives sample list and vocabulary
        def __init__(self, samples, vocab):
            # Save sample list
            self.samples = samples
            # Save vocabulary (reuse from training set)
            self.vocab = vocab
        # Return total number of samples
        def __len__(self): return len(self.samples)
        # Get sample by index, convert characters to vocabulary indices
        def __getitem__(self, idx):
            # Get text and label
            text, label = self.samples[idx]
            # Convert characters to index list
            x = [self.vocab.get(ch, self.vocab["<UNK>"]) for ch in text]
            # Return tensorized input and label
            return torch.tensor(x, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    # Create validation dataset object using training set's vocabulary
    val_ds = TestDataset(val_samples, train_ds.vocab)
    # Create test dataset object using training set's vocabulary
    test_ds = TestDataset(test_samples, train_ds.vocab)
    # Create training data loader, set batch_size and shuffle=True (shuffle each epoch)
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    # Create validation data loader, do not shuffle data
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"])
    # Create test data loader, do not shuffle data
    test_loader = DataLoader(test_ds, batch_size=params["batch_size"])
    # Build model
    # Get vocabulary size
    vocab_size = len(train_ds.vocab)
    # Instantiate model class, pass in various parameters
    model = model_cls(
        vocab_size=vocab_size,
        embed_dim=params["embed_dim"],
        hidden_dim=params["hidden_dim"],
        class_amount=5,  # 5 positions correspond to 5 classes
        num_layers=params.get("num_layers", 1),  # Get number of layers parameter, default 1
        dropout=params.get("dropout", 0.0),  # Get dropout parameter, default 0
    ).to(device)  # Move model to specified device
    # Define cross-entropy loss function (suitable for multi-class classification task)
    criterion = nn.CrossEntropyLoss()
    # Define Adam optimizer, set learning rate
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    # Training
    # Initialize best validation accuracy to 0
    best_val_acc = 0
    # Initialize best model state to None
    best_state = None
    # Loop training for the specified number of epochs
    for epoch in range(1, params["epochs"] + 1):
        # Train for one epoch, return average loss and accuracy
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # Evaluate on validation set, return average loss and accuracy
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        # If current validation accuracy is better than historical best, save model state
        if val_acc > best_val_acc:
            # Update best validation accuracy
            best_val_acc = val_acc
            # Deep copy current model state dictionary (use clone to ensure independent copy)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        # Print training info every 5 epochs or on the first epoch
        if epoch % 5 == 0 or epoch == 1:
            # Print epoch number, training loss, training accuracy, validation loss, validation accuracy
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Accuracy: {train_acc:.4f} | "
                  f"Validation Loss: {val_loss:.4f} Accuracy: {val_acc:.4f}")
    # Test set evaluation
    # Load model parameters that performed best on validation set
    model.load_state_dict(best_state)
    # Evaluate best model on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    # Print test set accuracy
    print(f"\n  *** Test Accuracy: {test_acc:.4f} ***\n")
    # Return test set accuracy
    return test_acc
# Execute the following code if this script is run as the main program (not imported)
if __name__ == "__main__":
    # Set PyTorch random seed for reproducibility
    torch.manual_seed(42)
    # Set NumPy random seed for reproducibility
    np.random.seed(42)
    # Define hyperparameters shared by all models
    common_params = {
        "batch_size": 64,       # 64 samples per batch
        "epochs": 20,           # Train for 20 epochs
        "embed_dim": 32,        # Character embedding vector dimension is 32
        "hidden_dim": 64,       # RNN/LSTM hidden layer dimension is 64
        "lr": 0.01,             # Learning rate is 0.01
    }
    # Run RNN experiment, using shared parameters
    rnn_acc = run_experiment("RNN", RNNClassifier, {**common_params})
    # Run LSTM experiment, using shared parameters with additional settings: 2 layers and 0.2 dropout
    lstm_acc = run_experiment("LSTM", LSTMClassifier, {**common_params, "num_layers": 2, "dropout": 0.2})
    # Print separator line
    print(f"\n{'='*60}")
    # Print results summary title
    print("  Experimental Results Summary")
    # Print separator line
    print(f"{'='*60}")
    # Print RNN test accuracy
    print(f"  RNN  Test Accuracy: {rnn_acc:.4f}")
    # Print LSTM test accuracy
    print(f"  LSTM Test Accuracy: {lstm_acc:.4f}")
    # Print separator line
    print(f"{'='*60}")
