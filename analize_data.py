import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Class mapping for labels
classes = {0: "attm",
           1: "ctrw",
           2: "fbm",
           3: "lw",
           4: "sbm"}


def plot_trajectory(trajectory: np.ndarray, title: str = "Trajectory") -> None:
    """
    Plots the trajectory in 2D space with markers for the start and end points.

    Args:
    :argument: trajectory (np.ndarray): The trajectory data to plot. It should be a Nx2 array where N is the number of
                points.
    :argument: title (str, optional): The title for the plot. Default is "Trajectory".

    Returns:
    :return: None
    """
    plt.figure(figsize=(5, 5))
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', markersize=2, label="Trajectory")
    plt.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=30, label="Start")
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=30, label="End")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()


def check_class_balance(path: str, title: str = "Class distribution in dataset") -> None:
    """
    Checks and visualizes the class distribution in the dataset.

    Args:
    :argument: path (str): Path to the file containing the class labels in one-hot encoded format (e.g., .npy file).
    :argument: title (str, optional): The title for the class distribution plot. Default is "Class distribution in
                dataset".

    Returns:
    :return: None
    """
    labels = np.load(path)  # Load the one-hot encoded labels
    y_train_classes = np.argmax(labels, axis=1)  # Convert to class indices
    class_counts = pd.Series(y_train_classes).value_counts()  # Get class counts

    plt.figure(figsize=(8, 6))
    bars = plt.bar(class_counts.index, class_counts.values, color='skyblue', alpha=0.8)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 5,
                 str(bar.get_height()),
                 ha='center', va='bottom', fontsize=10)
    plt.title(title, fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(class_counts.index, [classes[i] for i in class_counts.index])
    plt.show()


def plot_each_class_examples(x_train_path: str, y_train_path: str, quantity: int) -> None:
    """
    Plots a specified number of example trajectories for each class.

    Args:
    :argument: x_train_path (str): Path to the input features (trajectories).
    :argument: y_train_path (str): Path to the corresponding class labels.
    :argument: quantity (int): The number of examples to plot for each class.

    Returns:
    :return: None
    """
    x_train = np.load(x_train_path)  # Load the input data
    y_train = np.load(y_train_path)  # Load the class labels
    y_classes = np.argmax(y_train, axis=1)  # Convert one-hot encoded labels to class indices

    for class_label in range(len(classes)):
        class_indices = np.where(y_classes == class_label)[0]  # Get indices for the current class

        for i in range(min(quantity, len(class_indices))):
            trajectory = x_train[class_indices[i]]  # Get trajectory for the example
            plot_trajectory(trajectory, title=f"Class {classes.get(class_label)} - Example {i + 1}")
