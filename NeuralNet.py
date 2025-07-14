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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import itertools


class NeuralNet:
    def __init__(self, id, header=True):

        self.processed_data = None

        print("---- Loading Concrete Compressive Strength Dataset ----")
        try:
            concrete_data = fetch_ucirepo(id=id)
            self.raw_input = concrete_data
        except Exception as e:
            print("Error loading Concrete Strength Dataset: " + str(e))
            return




    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        print("---- Preprocessing Concrete Strength Dataset ----")

        # Extracting features and targets from the dataset to Pandas DataFrame
        X = self.raw_input.data.features # Features
        y = self.raw_input.data.targets # Target


        X_cleaned, Y_cleaned = clean_data(X, y)

        self.processed_data = combined_df = pd.concat([X_cleaned, Y_cleaned], axis=1)

        return 0

    # TODO: Train and evaluate models for all combinations of parameters
    # specified in the init method. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        print("---- Training and Evaluating Neural Network Models ----")

        # Extract features and targets from processed data
        X = self.processed_data.iloc[:, :-1]  # All columns except the last one (features)
        y = self.processed_data.iloc[:, -1]   # Last column (target)

        #Split Data into Training and Testing Sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Below are the hyperparameters that you need to use for model evaluation
        # You can assume any fixed number of neurons for each hidden layer. 
        
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]
        num_neurons = [10, 50]  # Adding the num_neurons parameter

        # Store results for each model
        model_results = []

        # Create all combinations of hyperparameters
        param_combinations = list(itertools.product(
            activations, learning_rate, max_iterations, num_hidden_layers, num_neurons
        ))

        print(f"Training {len(param_combinations)} different model configurations...")

        # Setup the plot for model histories
        plt.figure(figsize=(15, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, len(param_combinations)))

        for i, (activation, lr, max_iter, n_layers, n_neurons) in enumerate(param_combinations):
            print(f"\nModel {i+1}/{len(param_combinations)}: "
                  f"activation={activation}, lr={lr}, max_iter={max_iter}, "
                  f"layers={n_layers}, neurons={n_neurons}")

            # Create hidden layer sizes tuple
            hidden_layer_sizes = tuple([n_neurons] * n_layers)

            # Create and train the MLPRegressor
            mlp = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                learning_rate_init=lr,
                max_iter=max_iter,
                random_state=42,
                early_stopping=False,
                validation_fraction=0.0  # Disable validation split for loss curve
            )

            # Train the model
            mlp.fit(X_train_scaled, y_train)

            # Make predictions
            y_train_pred = mlp.predict(X_train_scaled)
            y_test_pred = mlp.predict(X_test_scaled)

            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            # Store results
            model_result = {
                'model_id': i+1,
                'activation': activation,
                'learning_rate': lr,
                'max_iterations': max_iter,
                'num_hidden_layers': n_layers,
                'num_neurons': n_neurons,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'n_iterations': mlp.n_iter_
            }
            model_results.append(model_result)

            # Plot loss curve (training history)
            if hasattr(mlp, 'loss_curve_'):
                epochs = range(1, len(mlp.loss_curve_) + 1)
                model_label = f"Model {i+1}: {activation}, lr={lr}, iter={max_iter}, layers={n_layers}, neurons={n_neurons}"
                plt.plot(epochs, mlp.loss_curve_, color=colors[i], label=model_label, linewidth=2)

            # Print model performance
            print(f"  Training MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
            print(f"  Test MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
            print(f"  Converged in {mlp.n_iter_} iterations")

        # Finalize the plot
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss (MSE)')
        plt.title('Training Loss Curves for All Neural Network Models')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Print summary of all models
        print("\n" + "="*80)
        print("SUMMARY OF ALL MODELS")
        print("="*80)

        results_df = pd.DataFrame(model_results)
        print("\nDetailed Results:")
        print(results_df.to_string(index=False))

        # Find best models
        best_train_model = results_df.loc[results_df['train_r2'].idxmax()]
        best_test_model = results_df.loc[results_df['test_r2'].idxmax()]

        print(f"\nBest Training Performance (R² = {best_train_model['train_r2']:.4f}):")
        print(f"  Model {best_train_model['model_id']}: {best_train_model['activation']}, "
              f"lr={best_train_model['learning_rate']}, iter={best_train_model['max_iterations']}, "
              f"layers={best_train_model['num_hidden_layers']}, neurons={best_train_model['num_neurons']}")

        print(f"\nBest Test Performance (R² = {best_test_model['test_r2']:.4f}):")
        print(f"  Model {best_test_model['model_id']}: {best_test_model['activation']}, "
              f"lr={best_test_model['learning_rate']}, iter={best_test_model['max_iterations']}, "
              f"layers={best_test_model['num_hidden_layers']}, neurons={best_test_model['num_neurons']}")

        return model_results

def clean_data(X, y):
    print("\n--- Cleaning Data ---")
    # Variables to be excluded for optimization
    features_to_exclude = ['Fly Ash','Coarse Aggregate', 'Fine Aggregate']
    X_filtered = X.drop(columns=features_to_exclude)

    # Combine X_filtered(features) and y(target) temporarily for unified cleaning based on index
    combined_df = pd.concat([X_filtered, y], axis=1)

    print("--- Original X and y Shapes ---")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("\n--- Initial Combined DataFrame Info ---")
    combined_df.info()

    # Handling Null Values
    print("\n--- Checking for null values in the dataset, if present remove ---")
    null_values = combined_df.isnull().sum()
    if null_values.sum() > 0:
        print("\n--- Dropping null Values ---")
        combined_df.dropna(inplace=True)

    # Handling Outliers using IQR (Winsorization/Capping)
    print("\n--- Outlier Handling (Winsorization using IQR method) ---")

    # Features to apply outlier handling
    features_to_process = X_filtered.columns

    # Calculate Q1, Q3, and IQR for each numerical column within these specific features
    Q1 = combined_df[features_to_process].quantile(0.25)
    Q3 = combined_df[features_to_process].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Apply Winsorization to the X part of combined_df
    for column in features_to_process:  # Iterate through the feature column names
        # Cap values below the lower bound for the current column
        combined_df[column] = np.where(
            combined_df[column] < lower_bound[column],  # Access the specific bound using column name
            lower_bound[column],
            combined_df[column]
        )
        # Cap values above the upper bound for the current column
        combined_df[column] = np.where(
            combined_df[column] > upper_bound[column],
            upper_bound[column],
            combined_df[column]
        )

    print("\n--- Combined DataFrame after Winsorization (Outlier Capping) ---")
    print(combined_df[features_to_process].describe())

    # Checking for Duplicates in the data
    print("\n--- Duplicate Rows Check ---")
    duplicate_data = combined_df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_data}")

    if duplicate_data > 0:
        print(f"Removing {duplicate_data} duplicate rows from combined_df...")
        combined_df_cleaned = combined_df.drop_duplicates()
        print(f"Shape after removing duplicates: {combined_df_cleaned.shape}")
    else:
        combined_df_cleaned = combined_df.copy()
        print("No duplicate data found.")

    # Separate X and y again from the cleaned combined_df
    X_cleaned = combined_df_cleaned[X_filtered.columns]
    y_cleaned = combined_df_cleaned[y.columns]

    # --- Final Cleaned DataFrames ---
    print("\n--- Cleaned Features Info ---")
    print("\t--- Information Extraction ---")
    X_cleaned.info()
    print(f"\t--- Descriptive Statistics ---\n{X_cleaned.describe()}")

    print("\n--- Cleaned Target Info ---")
    print("\t--- Information Extraction ---")
    y_cleaned.info()
    print(f"\t--- Descriptive Statistics ---\n{y_cleaned.describe()}")

    return X_cleaned, y_cleaned



if __name__ == "__main__":
    neural_network = NeuralNet(165) # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()
