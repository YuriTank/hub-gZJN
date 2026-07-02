import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.optim as optim
import torch.nn.functional as F

from src import Preprocessing
from src import TweetClassifier

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src import parameter_parser


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class DatasetMaper(Dataset):
    """
    Handles batches of dataset
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Execute:
    """
    Class for execution. Initializes the preprocessing as well as the
    Tweet Classifier model
    """

    def __init__(self, args):
        self.loader_test = None
        self.loader_training = None
        self.__init_data__(args)

        self.args = args
        self.batch_size = args.batch_size
        self.device = get_device()

        self.model = TweetClassifier(args).to(self.device)

    def __init_data__(self, args):
        """
        Initialize preprocessing from raw dataset to dataset split into training and testing
        Training and test datasets are index strings that refer to tokens
        """
        self.preprocessing = Preprocessing(args)
        self.preprocessing.load_data()
        self.preprocessing.prepare_tokens()

        raw_x_train = self.preprocessing.x_train
        raw_x_test = self.preprocessing.x_test

        self.y_train = self.preprocessing.y_train
        self.y_test = self.preprocessing.y_test

        self.x_train = self.preprocessing.sequence_to_token(raw_x_train)
        self.x_test = self.preprocessing.sequence_to_token(raw_x_test)

    def train(self):

        training_set = DatasetMaper(self.x_train, self.y_train)
        test_set = DatasetMaper(self.x_test, self.y_test)

        self.loader_training = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        self.loader_test = DataLoader(test_set)

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=True
        )

        best_test_accuracy = 0.0
        best_model_state = None
        patience_counter = 0
        early_stop_patience = 15

        for epoch in range(self.args.epochs):

            train_predictions = []
            train_labels = []
            epoch_loss = 0.0
            num_batches = 0

            self.model.train()

            for x_batch, y_batch in self.loader_training:

                x = x_batch.type(torch.LongTensor).to(self.device)
                y = y_batch.type(torch.FloatTensor).view(-1, 1).to(self.device)

                y_pred = self.model(x)

                loss = F.binary_cross_entropy(y_pred, y)

                optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                train_predictions += list(y_pred.squeeze(1).detach().cpu().numpy())
                train_labels += list(y.squeeze(1).detach().cpu().numpy())

            test_predictions = self.evaluation()

            train_accuracy = self.calculate_accuracy(train_labels, train_predictions)
            test_accuracy = self.calculate_accuracy(self.y_test, test_predictions)

            print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (epoch+1, epoch_loss / num_batches, train_accuracy, test_accuracy))

            scheduler.step(test_accuracy)

            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print("Early stopping at epoch %d (best test accuracy: %.5f)" % (epoch+1, best_test_accuracy))
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("Loaded best model with test accuracy: %.5f" % best_test_accuracy)

    def evaluation(self):

        predictions = []
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in self.loader_test:
                x = x_batch.type(torch.LongTensor).to(self.device)

                y_pred = self.model(x)
                predictions += list(y_pred.squeeze(1).detach().cpu().numpy())

        return predictions

    @staticmethod
    def calculate_accuracy(grand_truth, predictions):
        true_positives = 0
        true_negatives = 0

        for true, pred in zip(grand_truth, predictions):
            if (pred > 0.5) and (true == 1):
                true_positives += 1
            elif (pred < 0.5) and (true == 0):
                true_negatives += 1
            else:
                pass

        return (true_positives+true_negatives) / len(grand_truth)

if __name__ == "__main__":

    args = parameter_parser()

    execute = Execute(args)
    execute.train()
