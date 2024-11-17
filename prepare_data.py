import numpy as np
from sklearn.model_selection import train_test_split


def merge_data(x_train, x_val, y_train, y_val, path_to_dir):
    X_total = np.concatenate([x_train, x_val], axis=0)
    y_total = np.concatenate([y_train, y_val], axis=0)
    np.save(path_to_dir + "/X_total.npy", X_total)
    np.save(path_to_dir + "/y_total.npy", y_total)


def split_dataset_stratified(x_path, y_path, path='final_data', train_size=0.7, val_size=0.15, test_size=0.15,
                             random_state=None):
    """
    Splits the dataset into training, validation, and test sets while preserving class proportions. Saves them to files.

    :param x_path: Path to the .npy file with X_total data
    :param y_path: Path to the .npy file with Y_total data
    :param train_size: Proportion of the training set
    :param val_size: Proportion of the validation set
    :param test_size: Proportion of the test set
    :param random_state: Random seed for reproducibility
    :param path: Path where sets should be stored
    :return: None
    """
    # Ensure proportions sum up to 1
    assert np.isclose(train_size + val_size + test_size, 1.0), "The sum of proportions must be 1."

    # Load the data
    X_total = np.load(x_path)
    Y_total = np.load(y_path)

    # Convert one-hot labels to class indices
    Y_classes = np.argmax(Y_total, axis=1)

    # Split into training and temporary sets (validation + test)
    X_train, X_temp, Y_train, Y_temp, Y_train_classes, Y_temp_classes = train_test_split(
        X_total, Y_total, Y_classes,
        test_size=(1 - train_size), stratify=Y_classes, random_state=random_state
    )

    # Calculate the ratio of validation and test sets within the temporary set
    val_test_ratio = val_size / (val_size + test_size)

    # Split the temporary set into validation and test sets
    X_val, X_test, Y_val, Y_test, Y_val_classes, Y_test_classes = train_test_split(
        X_temp, Y_temp, Y_temp_classes,
        test_size=(1 - val_test_ratio), stratify=Y_temp_classes, random_state=random_state
    )

    np.save(path + "/X_train.npy", X_train)
    np.save(path + "/y_train.npy", Y_train)
    np.save(path + "/X_val.npy", X_val)
    np.save(path + "/y_val.npy", Y_val)
    np.save(path + "/X_test.npy", X_test)
    np.save(path + "/y_test.npy", Y_test)
