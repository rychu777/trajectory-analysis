import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Bidirectional, Attention, LayerNormalization
from sklearn.preprocessing import label_binarize


class NeuralNetworkClassifier:
    def __init__(self, path_x_train: str, path_y_train: str, path_x_val: str, path_y_val: str, path_x_test: str,
                 path_y_test: str) -> None:
        """
        Initialize the NeuralNetworkClassifier.

        Args:
        :argument: path_x_train (str): Path to the features of training dataset.
        :argument: path_y_train (str): Path to the labels of training dataset.
        :argument: path_x_val (str): Path to the features validation dataset.
        :argument: path_y_val (str): Path to the labels of validation dataset.
        :argument: path_x_test (str): Path to the features of testing dataset.
        :argument: path_y_test (str): Path to the labels of testing dataset.

        Returns:
        :return: None
        """
        self.x_train = np.load(path_x_train)
        print(self.x_train.shape)
        self.y_train = np.load(path_y_train)

        self.x_val = np.load(path_x_val)
        print(self.x_val.shape)
        self.y_val = np.load(path_y_val)

        self.x_test = np.load(path_x_test)
        self.y_test = np.load(path_y_test)
        print(self.x_test.shape)
        self.model = None
        self.history = None

    def build_model(self):
        """
        Build the neural network model with a modified architecture for diffusion classification.
        """
        input_layer = Input(shape=(300, 2))  # 300 time steps, 2 features (x and y coordinates)

        x = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
        x = Dropout(0.3)(x)

        # Add another LSTM layer to capture more complex patterns
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.3)(x)

        # Apply Attention Mechanism (Self-Attention) here
        query = x  # LSTM output serves as both query and value
        value = x
        key = x  # Self-attention mechanism works with these 3 inputs
        attention_output = Attention()([query, value, key])

        attention_output = LayerNormalization()(attention_output)

        # Further processing of the attention output with another LSTM layer
        x = LSTM(32)(attention_output)
        x = Dropout(0.3)(x)

        # Dense layers for final classification
        x = Dense(64, activation='relu')(x)

        # Output layer for 5 classes (types of diffusion)
        output_layer = Dense(5, activation='softmax')(x)  # 5 diffusion types as classes

        # Compile the model with an Adam optimizer and categorical cross-entropy loss function
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.summary()

    def train(self, epochs: int = 20, batch_size: int = 64) -> None:
        """
        Train the neural network model.

        Args:
        :argument: epochs (int): Number of epochs for training.
        :argument: batch_size (int): Batch size for training.

        Returns:
        :return: None
        """
        if self.model is None:
            self.build_model()

        self.model.fit(x=self.x_train, y=self.y_train, epochs=epochs, batch_size=batch_size, validation_data=
                        (self.x_val, self.y_val), verbose=1)

    def predict(self) -> np.ndarray:
        """
        Make predictions using the trained model.

        Returns:
        :return: np.ndarray: Predicted values.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        return self.model.predict(self.x_test)

    def evaluate(self, y_pred: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
        """
        Evaluate the performance of the model.

        Args:
        :argument: y_pred (np.ndarray): Predicted values.

        Returns:
        :return: tuple: Accuracy, precision, recall, F1-score, and confusion matrix.
        """
        # Convert predicted probabilities to class labels
        y_pred_classes = np.argmax(y_pred, axis=1)  # Take the class with the highest probability

        # Convert the true labels from one-hot encoding if necessary
        y_true_classes = np.argmax(self.y_test, axis=1)  # If y_test is one-hot encoded

        accuracy = accuracy_score(y_true_classes, y_pred_classes)

        # Specify 'average' to handle multi-class classification
        precision = precision_score(y_true_classes, y_pred_classes,
                                    average='weighted')  # You can also use 'micro', 'macro', etc.
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

        return accuracy, precision, recall, f1, conf_matrix

    def plot_roc_curve(self, y_pred: np.ndarray) -> None:
        """
        Plot the ROC curve for multiclass classification.

        Args:
        :argument: y_pred (np.ndarray): Predicted values (probabilities for each class).
        Returns:
        :return: None
        """
        # Binarize the output labels for multiclass classification (one-vs-rest)
        y_test_bin = label_binarize(self.y_test, classes=[0, 1, 2, 3, 4])
        n_classes = y_test_bin.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
            roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_pred[:, i])

        plt.figure(figsize=(10, 8))

        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'r--', label='Random: AUC = 0.5')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multiclass ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig('resources/ROC.png')
        plt.show()
