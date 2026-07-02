import argparse

def parameter_parser():

     parser = argparse.ArgumentParser(description = "Tweet Classification")

     parser.add_argument("--epochs",
                                dest = "epochs",
                                type = int,
                                default = 100,
                         help = "Number of gradient descent iterations. Default is 100.")

     parser.add_argument("--learning_rate",
                                dest = "learning_rate",
                                type = float,
                                default = 0.0005,
                         help = "Gradient descent learning rate. Default is 0.0005.")

     parser.add_argument("--hidden_dim",
                                dest = "hidden_dim",
                                type = int,
                                default = 64,
                         help = "Number of neurons by hidden layer. Default is 64.")

     parser.add_argument("--lstm_layers",
                                dest = "lstm_layers",
                                type = int,
                                default = 1,
                     help = "Number of LSTM layers")

     parser.add_argument("--batch_size",
                                    dest = "batch_size",
                                    type = int,
                                    default = 64,
                             help = "Batch size")

     parser.add_argument("--test_size",
                                dest = "test_size",
                                type = float,
                                default = 0.20,
                         help = "Size of test dataset. Default is 10%.")

     parser.add_argument("--max_len",
                                dest = "max_len",
                                type = int,
                                default = 30,
                         help = "Maximum sequence length per tweet")

     parser.add_argument("--max_words",
                                dest = "max_words",
                                 type = int,
                                default = 2000,
                         help = "Maximum number of words in the dictionary")

     parser.add_argument("--data_path",
                                 dest = "data_path",
                                 type = str,
                                 default = None,
                          help = "Optional path to the tweets CSV file.")
     return parser.parse_args()
