#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# fetch dataset 
real_estate_valuation = fetch_ucirepo(id=477) 
  
# data (as pandas dataframes) 


class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = dataFile




    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    
    def preprocess(self):
        
        
        features = self.raw_input.data.features 
        actual_value = self.raw_input.data.targets 
        features = (features - features.mean()) / features.std() #this is called normalization, ensures all features are on the same scale
        print("Checking for duplicate rows in features")
        print(features.duplicated().sum())

        features = features.drop_duplicates()
        actual_value = actual_value.loc[features.index]
        self.processed_data = pd.concat([features, actual_value], axis=1)

        return 0

    # TODO: Train and evaluate models for all combinations of parameters
    # specified in the init method. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y)

        # Below are the hyperparameters that you need to use for model evaluation
        # You can assume any fixed number of neurons for each hidden layer. 
        
        activations = ['logistic', 'tanh', 'relu']
        learning_rates = [0.01, 0.1]
        max_iterations = [5000, 10000] # also known as epochs
        num_hidden_layers = [2, 3]


        results = []

        for activation in activations:
            for learning_rate in learning_rates:
                for max_iter in max_iterations:
                    for num_layers in num_hidden_layers:
                        model = MLPRegressor(hidden_layer_sizes=(num_layers,),
                                            activation=activation,
                                            learning_rate_init=learning_rate,
                                            max_iter=max_iter,
                                            random_state=42)
                        model.fit(X_train, y_train)

                        train_accuracy = model.score(X_train, y_train)
                        test_accuracy = model.score(X_test, y_test)

                        # Track or plot model history (accuracy vs epochs) if needed

                        results.append({
                            'Activation': activation,
                            'Learning Rate': learning_rate,
                            'Max Iterations': max_iter,
                            'Num Hidden Layers': num_layers,
                            'Train Accuracy': train_accuracy,
                            'Test Accuracy': test_accuracy,
                            'Model': model
                        })

    # Print or return results table, plot model history, etc.
        resulttable = pd.DataFrame(results)
        print(resulttable)
        # Create the neural network and be sure to keep track of the performance
        #   metrics

        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.
        plt.figure(figsize=(10, 6))
        for i, result in enumerate(results):
            # Ensure result is a dictionary with the expected keys
            plt.plot(result['Model'].loss_curve_, label=f"Model {i + 1}: {result['Activation']} - {result['Num Hidden Layers']} layers")

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss vs. Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    neural_network = NeuralNet(real_estate_valuation) # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()
