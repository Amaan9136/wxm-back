import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_error, confusion_matrix
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from .mul_cmd_models import MulCmdModels

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend/routes/store_mulcmd_data/uploads/')

class MulCmd:
    def __init__(self):
        self.data = None
        self.features = None
        self.target = None
        self.model = None
        self.model_name = None
        self.X_train, self.X_test, self.y_train, self.y_test = (None, None, None, None)
        self.feature_names = None
        self.target_names = None
        self.predictions = None
        self.model_manager = MulCmdModels()
        self.label_encoders = {}

    def file(self, filenames):
        output_messages = []
        for filename in filenames:
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.exists(file_path):
                self.data = pd.read_csv(file_path)
                self.feature_names = self.data.columns.tolist()
                output_messages.append(f"Data loaded from '{filename}'")
            else:
                output_messages.append(f"File '{filename}' does not exist.")
        return "\n".join(output_messages)
    
    def encode_features(self, encoding_command='label= label_columns onehot= onehot_columns'):
        # THERE IS AN ISSUE USING THIS, IF THE DATA SET IS ENCODED 
        # THEN THE FEATURES ARE SET AS COMPLETE DATASET INSTED OF ONLY SETTING THE FEATURE FEATURE_RANGE AND TARGET TARGET_RANGE
        if self.data is None:
            return "No data loaded. Please load the file first."

        label_columns = []
        onehot_columns = []

        label_pattern = re.compile(r'label\s*=?\s*([\d,]*)')
        onehot_pattern = re.compile(r'onehot\s*=?\s*([\d,]*)')

        label_match = label_pattern.search(encoding_command)
        onehot_match = onehot_pattern.search(encoding_command)

        if label_match:
            label_columns = list(map(int, label_match.group(1).split(','))) if label_match.group(1) else []
        if onehot_match:
            onehot_columns = list(map(int, onehot_match.group(1).split(','))) if onehot_match.group(1) else []

        print("Original Data Sample:")
        print(self.data.head())

        encoded_df = self.data.copy()
        feature_names = self.data.columns.tolist()

        if label_columns:
            for col in label_columns:
                if col < len(self.data.columns):
                    column_name = self.data.columns[col]
                    if self.data[column_name].dtype == 'object':
                        if column_name not in self.label_encoders:
                            self.label_encoders[column_name] = LabelEncoder()
                        encoded_df[column_name] = self.label_encoders[column_name].fit_transform(self.data[column_name])
                    else:
                        return f"Column '{column_name}' is not categorical and cannot be label-encoded."
                else:
                    return f"Column index {col} is out of range."

        print("\nData After Label Encoding Sample:")
        print(encoded_df.head())

        if onehot_columns:
            onehot_column_names = [self.data.columns[col] for col in onehot_columns if col < len(self.data.columns)]
            if onehot_column_names:
                preprocessor = ColumnTransformer(
                    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), onehot_column_names)],
                    remainder='passthrough'
                )
                encoded_array = preprocessor.fit_transform(encoded_df)
                
                feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(onehot_column_names))
                feature_names.extend([name for name in self.data.columns if name not in onehot_column_names])
                encoded_df = pd.DataFrame(encoded_array, columns=feature_names)

            else:
                return "No columns specified for One-Hot Encoding."

        print("\nFinal Encoded Data Sample:")
        print(encoded_df.head())

        head_str = encoded_df.head().to_string(index=False)

        self.features = encoded_df
        self.feature_names = feature_names
        return f"Columns encoded. Features:\n{head_str}"
    
    def features_range(self, start, end=None):
        if self.data is None:
            return "No data loaded. Please load the file first."

        if end is None:
            end = start + 1
        else:
            end += 1
        
        if 0 <= start < end <= self.data.shape[1]:
            self.features = self.data.iloc[:, start:end]
            self.feature_names = self.data.columns[start:end].tolist()

            if end - start == 1:
                return f"Feature set from column {start} ({self.feature_names[0]})."
            else:
                return f"Features set from column {start} to {end - 1} ({', '.join(self.feature_names)})."
        else:
            return f"Invalid feature range. Data has only {self.data.shape[1]} columns"

    def target_range(self, *columns):
        if self.data is None:
            return "No data loaded. Please load the file first."

        if len(columns) > 0:
            try:
                column_indices = [int(i) for i in columns]

                if all(0 <= i < self.data.shape[1] for i in column_indices):
                    if len(self.feature_names) != self.data.shape[1]:
                        self.feature_names = self.data.columns.tolist()
                    
                    self.target = self.data.iloc[:, column_indices]
                    self.target_names = [self.feature_names[i] for i in column_indices]
                    
                    if len(column_indices) == 1:
                        self.target = self.target.iloc[:, 0]
                        self.target_names = [self.target_names[0]]
                        return f"Target set to column {column_indices[0]} ({self.target_names[0]})."
                    else:
                        return f"Target set to columns {', '.join(map(str, column_indices))} ({', '.join(self.target_names)})."
                else:
                    return "Error: Target Column index out of range."

            except ValueError as ve:
                return f"Error: Invalid column index format. Ensure all indices are integers. Details: {str(ve)}"
            except IndexError as ie:
                return f"Error: Target Column index out of range. Details: {str(ie)}"
            except Exception as e:
                return f"Unexpected error: {str(e)}"

        else:
            return "No target columns specified."

    def split(self, ratio):
        if self.features is None or self.target is None:
            return "Features or target not set. Please specify both."

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=ratio, random_state=42)
        return f"Data split with ratio {ratio}."

    def set_model(self, model_name, **kwargs):
        # print("features: ",self.features,"target: ",self.target)
        model, message, X_train, y_train, X_test, y_test = self.model_manager.set_model(
            model_name, self.X_train, self.y_train, self.X_test, self.y_test, **kwargs
        )
        
        if model is None:
            return message
        
        self.model = model
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        trained_message = ""
        if self.X_train is not None and self.y_train is not None:
            trained_message = self.train()
        
        return f"{message} \n{trained_message}"

    def train(self):
        if self.model and self.X_train is not None and self.y_train is not None:
            self.model.fit(self.X_train, self.y_train)
            return "Model trained successfully."
        else:
            return "Model or training data is missing."

    def print_predict(self):
        if self.model and self.X_test is not None:
            self.predictions = self.model.predict(self.X_test)
            result = []

            # Convert X_test and y_test to 2D arrays
            X_test_array = np.array(self.X_test)
            y_test_array = np.array(self.y_test)

            predictions_list = self.predictions.tolist()

            result.append(f"Test Features: {X_test_array}")
            result.append(f"Actual values: {y_test_array}")
            result.append(f"Predictions: {predictions_list}")
            return "\n".join(result)
        else:
            return "Model or test data is missing to predict."


    def print_accuracy(self):
        if self.model and self.X_test is not None:
            self.predictions = self.model.predict(self.X_test)
            try:
                score = accuracy_score(self.y_test, self.predictions)
                cm = confusion_matrix(self.y_test, self.predictions)
                return f"Accuracy Score: {score}\nConfusion Matrix:\n{cm}"
            except ValueError:
                return "Accuracy cannot be computed for regression tasks."
        else:
            return "Model or test data is missing to find accuracy."

    def print_r2(self):
        if self.model and self.X_test is not None:
            self.predictions = self.model.predict(self.X_test)
            try:
                score = r2_score(self.y_test, self.predictions)
                return f"R2 Score: {score}"
            except ValueError:
                return "R2 Score cannot be computed for classification tasks."
        else:
            return "Model or test data is missing to find R2 Score."

    def print_confusion_matrix(self):
        if self.model and self.X_test is not None:
            self.predictions = self.model.predict(self.X_test)
            try:
                score = confusion_matrix(self.y_test, self.predictions)
                return f"Confusion Matrix: {score}"
            except ValueError:
                return "Confusion Matrix cannot be computed for regression tasks."
        else:
            return "Model or test data is missing to find Confusion Matrix."

    def print_mse(self):
        if self.model and self.X_test is not None:
            self.predictions = self.model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, self.predictions)
            return f"Mean Squared Error: {mse}"
        else:
            return "Model or test data is missing to find MSE."

    def print_mae(self):
        if self.model and self.X_test is not None:
            self.predictions = self.model.predict(self.X_test)
            mae = mean_absolute_error(self.y_test, self.predictions)
            return f"Mean Absolute Error: {mae}"
        else:
            return "Model or test data is missing to find MAE."

    def plot_data(self):
        if self.features is not None and self.target is not None:
            # Convert features to DataFrame if it's not already one
            if not isinstance(self.features, pd.DataFrame):
                features_df = pd.DataFrame(self.features)
            else:
                features_df = self.features

            num_features = features_df.shape[1]  # Number of features (columns)

            # Create subplots for each feature
            fig, axes = plt.subplots(num_features, 1, figsize=(10, 6 * num_features))
            if num_features == 1:
                axes = [axes]  # Handle the case with only one feature (axes would not be an array)

            for i in range(num_features):
                ax = axes[i]
                feature_name = self.feature_names[i]  # Get the column name

                # Plot the i-th feature against the target values
                ax.scatter(features_df.iloc[:, i], self.target, color='blue', alpha=0.6, s=100)
                ax.set_xlabel(f'{feature_name}')  # Use the column name as the label
                ax.set_ylabel('Target')
                ax.set_title(f'{feature_name} vs Target')
            
            plot_data_file = 'plot_data.png'
            plot_path = os.path.join(UPLOAD_FOLDER, plot_data_file)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

            return plot_data_file, f"Plot saved and retrieved as as '{plot_data_file}'"
        else:
            return "Features or target is not set for plotting."


    def plot_predict(self):
        if self.model and self.X_test is not None and self.y_test is not None:
            self.predictions = self.model.predict(self.X_test)

            # Convert X_test to a DataFrame if it's not already one
            if not isinstance(self.X_test, pd.DataFrame):
                X_test_df = pd.DataFrame(self.X_test)
            else:
                X_test_df = self.X_test

            num_features = X_test_df.shape[1]  # Number of features (columns)

            # Ensure that lengths of X_test, y_test, and predictions match
            if len(X_test_df) == len(self.y_test) == len(self.predictions):
                # Create subplots for each feature
                fig, axes = plt.subplots(num_features, 1, figsize=(10, 6 * num_features))
                if num_features == 1:
                    axes = [axes]  # To handle the case with only one feature (axes would not be an array)

                for i in range(num_features):
                    ax = axes[i]
                    feature_name = self.feature_names[i]  # Get the column name

                    # Plot the i-th feature from X_test against actual and predicted values
                    ax.scatter(X_test_df.iloc[:, i], self.y_test, color='blue', label='Actual', alpha=0.6, s=100)
                    ax.scatter(X_test_df.iloc[:, i], self.predictions, color='red', label='Predicted', alpha=0.6)
                    ax.set_xlabel(f'{feature_name}')  # Use the column name as the label
                    ax.set_ylabel('Target')
                    ax.set_title(f'{feature_name} vs Actual and Predicted')
                    ax.legend()

                plot_predict_file = 'plot_predict.png'
                plot_path = os.path.join(UPLOAD_FOLDER, plot_predict_file)
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()

                return plot_predict_file, f"Prediction plots saved and retrieved as as '{plot_predict_file}'"
            else:
                return "Lengths of X_test, y_test, and predictions are not the same."
        else:
            return "Model or test data is missing for prediction plotting."

    # def save_model(self, model_name):
    #     if self.model is not None:
    #         if not model_name.endswith('.pkl'):
    #             model_name += '.pkl'
    #         model_path = os.path.join(UPLOAD_FOLDER, model_name)
    #         with open(model_path, 'wb') as model_file:
    #             pickle.dump(self.model, model_file)
    #         return model_name, f"Model saved and retrieved as as '{model_name}'"
    #     else:
    #         return "No model to save."
    
    # def load_model(self, model_name):
    #     if not model_name.endswith('.pkl'):
    #         model_name += '.pkl'
    #     model_path = os.path.join(UPLOAD_FOLDER, model_name)

    #     if os.path.exists(model_path):
    #         with open(model_path, 'rb') as model_file:
    #             self.model = pickle.load(model_file)
    #         return f"Model loaded from '{model_path}'"
    #     else:
    #         return f"Model file '{model_path}' not found."

    # def model_predict(self, model_name, predict_data):
    #     self.load_model(model_name)

    #     if self.model is None:
    #         return "Failed to load model."
        
    #     try:
    #         if hasattr(self.model, 'feature_names_in_'):
    #             predict_data = pd.DataFrame(predict_data, columns=self.model.feature_names_in_)
    #         else:
    #             predict_data = np.array(predict_data)
            
    #         predictions = self.model.predict(predict_data)
    #         return predictions
    #     except Exception as e:
    #         return f"Error during prediction: {e}"