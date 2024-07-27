import numpy as np
import logging
import pandas as pd
import plotly.express as px
import plotly.io as pio
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import glorot_uniform
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch

# Configure logging to write to a file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('ann_regressor.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)


class ANNHyperModel(HyperModel):
    """
    ANNHyperModel class for hyperparameter tuning of an ANN regressor using Keras Tuner.

    This class defines the structure of the ANN model and specifies the hyperparameters to be tuned.

    Attributes:
        input_dim (int): Number of input features.
        output_dim (int): Number of output units. Default is 1.
        classification_type (str) : to handle both binary and multi-class classification. It defaults to binary.
    """

    def __init__(self, input_dim, output_dim=1, classification_type='binary'):
        """
        Initialize the ANNHyperModel.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output units. Default is 1.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.classification_type = classification_type

    def build(self, hp):
        """
        Build the ANN model with hyperparameters.

        Args:
            hp (HyperParameters): Hyperparameters for tuning.

        Returns:
            Sequential: Compiled Keras Sequential model.
        """
        model = Sequential()
        # Define the input layer
        # model.add(InputLayer(input_shape=(self.input_dim,)))
        # Tune the number of units in the first Dense layer
        units = hp.Int('units', min_value=32, max_value=512, step=32)
        # Define the first hidden layer with Glorot initialization
        # model.add(Dense(self.hidden_layers[0], activation='relu', kernel_initializer=glorot_uniform()))
        # Add input layer and first hidden layer
        model.add(Dense(units=units, activation='relu', input_dim=self.input_dim))
        # Tune the number of hidden layers and units in each layer
        for i in range(hp.Int('num_layers', 1, 5)):
            model.add(Dense(units=hp.Int('units_' + str(i),
                                         min_value=32,
                                         max_value=512,
                                         step=32),
                            activation='relu'))
            # model.add(Dense(units=hp.Int('units_' + str(i),
            #                              min_value=32,
            #                              max_value=512,
            #                              step=32),
            #                 activation='relu', kernel_initializer=glorot_uniform()))

            # Define the output layer
            if self.classification_type == 'binary':
                model.add(Dense(self.output_dim, activation='sigmoid'))
                # model.add(Dense(self.output_dim, activation='sigmoid', kernel_initializer=glorot_uniform()))
                loss = 'binary_crossentropy'
            else:  # for multi-class classification
                model.add(Dense(self.output_dim, activation='softmax'))
                # model.add(Dense(self.output_dim, activation='softmax', kernel_initializer=glorot_uniform()))
                loss = 'categorical_crossentropy'

        # Tune the learning rate for the optimizer
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])
        # model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss=loss, metrics=['AUC', 'accuracy'])
        return model


class ANNClassifier:
    """
    Artificial Neural Network (ANN) Classifier.

    This class encapsulates the creation, training, evaluation, and hyperparameter tuning
    of an ANN regressor model using TensorFlow and Keras.

    Attributes:
        input_dim (int): Number of input features.
        hidden_layers (list): List containing the number of units in each hidden layer.
        output_dim (int): Number of output units. Default is 1.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        model (Sequential): Keras Sequential model instance.
        history (History): Keras History object containing training history.
        classification_type (str) : to handle both binary and multi-class classification. It defaults to binary.
    """

    def __init__(self, input_dim, hidden_layers, output_dim=1, learning_rate=0.001, classification_type='binary'):
        """
        Initialize the ANN Classifier.

        Args:
            input_dim (int): Number of input features.
            hidden_layers (list): List containing the number of units in each hidden layer.
            output_dim (int): Number of output units. Default is 1.
            learning_rate (float): Learning rate for the optimizer. Default is 0.001.
            classification_type to handle both binary and multi-class classification. It defaults to binary.
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.classification_type = classification_type
        self.model = None
        self.history = None

    def build_model(self, learning_rate=None):
        """
        Build the ANN model.

        Args:
            learning_rate (float): Learning rate for the optimizer. If None, use self.learning_rate.
        """
        try:
            if learning_rate is None:
                learning_rate = self.learning_rate

            model = Sequential()
            # Define the input layer
            # model.add(InputLayer(input_shape=(self.input_dim,)))
            # Define the first hidden layer with Glorot initialization
            # model.add(Dense(self.hidden_layers[0], activation='relu', kernel_initializer=glorot_uniform()))
            # Adds the input layer and the first hidden layer
            model.add(Dense(self.hidden_layers[0], input_dim=self.input_dim, activation='relu'))

            # Add subsequent hidden layers
            for units in self.hidden_layers[1:]:
                model.add(Dense(units, activation='relu'))
                # model.add(Dense(units, activation='relu', kernel_initializer=glorot_uniform()))

            # Define the output layer
            if self.classification_type == 'binary':
                model.add(Dense(self.output_dim, activation='sigmoid'))
                # model.add(Dense(self.output_dim, activation='sigmoid', kernel_initializer=glorot_uniform()))
                loss = 'binary_crossentropy'
            else:  # for multi-class classification
                model.add(Dense(self.output_dim, activation='softmax'))
                # model.add(Dense(self.output_dim, activation='softmax', kernel_initializer=glorot_uniform()))
                loss = 'categorical_crossentropy'

            # Compile the model
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])  # uses stochastic gradient descent
            #model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])

            self.model = model
            logger.info("Model built successfully.")
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise

    def train(self, X_train, y_train, epochs=100, batch_size=32, verbose=1, learning_rate=None):
        """
        Train the ANN model.

        Args:
            X_train (numpy.ndarray): Training data.
            y_train (numpy.ndarray): Training labels.
            epochs (int): Number of epochs to train the model. Default is 100.
            batch_size (int): Batch size for training. Default is 32.
            verbose (int): Verbosity mode. Default is 1.
            learning_rate (float): Learning rate for the optimizer. If None, use self.learning_rate.
        """
        try:
            if self.model is None:
                self.build_model(learning_rate)
            self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
            logger.info("Model trained successfully.")
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def evaluate(self, X_test, y_test):
        """
        Evaluate the ANN model.

        Args:
            X_test (numpy.ndarray): Test data.
            y_test (numpy.ndarray): Test labels.

        Returns:
            float: Loss value.
        """
        try:
            if self.model is None:
                raise ValueError("Model is not built yet. Train the model first.")
            loss = self.model.evaluate(X_test, y_test)
            logger.info("Model evaluated successfully.")
            return loss
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise

    def predict(self, X):
        """
        Predict using the ANN model.

        Args:
            X (numpy.ndarray): Data for prediction.

        Returns:
            numpy.ndarray: Predicted values.
        """
        try:
            if self.model is None:
                raise ValueError("Model is not built yet. Train the model first.")
            predictions = self.model.predict(X)
            logger.info("Prediction made successfully.")
            return predictions
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise

    def tune(self, X_train, y_train, max_trials=3, executions_per_trial=1, directory='my_dir',
             project_name='ann_tuning'):
        """
        Tune hyperparameters of the ANN model using Keras Tuner.

        Args:
            X_train (numpy.ndarray): Training data.
            y_train (numpy.ndarray): Training labels.
            max_trials (int): Total number of trials (model configurations) to test. Default is 10.
            executions_per_trial (int): Number of models to train and evaluate for each trial. Default is 1.
            directory (str): Path to the directory to save the results. Default is 'my_dir'.
            project_name (str): Name to use as prefix for files saved by Keras Tuner. Default is 'ann_tuning'.

        Returns:
            RandomSearch: Keras Tuner RandomSearch instance after tuning.
        """
        try:
            hypermodel = ANNHyperModel(input_dim=self.input_dim, output_dim=self.output_dim,
                                       classification_type=self.classification_type)
            tuner = RandomSearch(
                hypermodel,
                objective='val_loss',
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory=directory,
                project_name=project_name
            )

            tuner.search(X_train, y_train, epochs=100, validation_split=0.2)
            self.model = tuner.get_best_models(num_models=1)[0]
            logger.info("Hyperparameter tuning completed successfully.")
            return tuner
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {e}")
            raise

    def plot_loss_history(self, filename='loss_history.png'):
        """
        Plot the training loss history and save it as a PNG file.

        Args:
            filename (str): The filename to save the plot as. Default is 'loss_history.png'.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        try:
            if self.history is None:
                raise ValueError("No training history found. Train the model first.")

            history_df = pd.DataFrame(self.history.history)
            fig = px.line(history_df, y='loss', title='Training Loss Over Epochs')
            pio.write_image(fig, filename, format='png')
            logger.info(f"Loss history plot saved successfully as {filename}.")
        except Exception as e:
            logger.error(f"Error plotting loss history: {e}")
            raise

    def model_summary(self):
        """
        Show ANN Model summary

        """
        try:
            print(self.model.summary())
            logger.info("Model Summary successfully.")
        except Exception as e:
            logger.error(f"Error printing model Summary: {e}")
            raise

    def __str__(self):
        """
        Custom string representation of the ANNRegressor.

        Returns:
            str: String representation of the ANNRegressor instance.
        """
        return (f'ANNRegressor(input_dim={self.input_dim}, hidden_layers={self.hidden_layers}, '
                f'output_dim={self.output_dim}, classification_type={self.classification_type} ,learning_rate={self.learning_rate})')


# Example usage

if __name__ == "__main__":
    # Sample data
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    X_test = np.random.rand(20, 10)
    y_test = np.random.rand(20)
    # Example usage for binary classification
    # ann = ANNModel(input_dim=10, hidden_layers=[64, 32], output_dim=1, classification_type='binary')
    # ann.build_model()

    # Example usage for multi-class classification
    # ann = ANNModel(input_dim=10, hidden_layers=[64, 32], output_dim=3, classification_type='multi-class')
    # ann.build_model()

    # Evaluate the model
    # loss = ann.evaluate(X_test, y_test)
    # print(f'Test Loss: {loss}')

    # Hyperparameter tuning
    # tuner = ann.tune(X_train, y_train, max_trials=10, executions_per_trial=1)
    # print(f'Best hyperparameters: {tuner.get_best_hyperparameters(num_trials=1)[0].values}')
