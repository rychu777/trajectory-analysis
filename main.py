from analize_data import *
from prepare_data import *
from model import *

if __name__ == '__main__':

    # merge_data(X_train,X_val,y_train,y_val, "total_data")
    # X_total = np.load('total_data/X_total.npy')
    # y_total = np.load('total_data/y_total.npy')

    # print(f"X_total shape: {X_total.shape}, y_total shape: {y_total.shape}")

    # split_dataset_stratified("total_data/X_total.npy", "total_data/y_total.npy", train_size=0.7, val_size=0.15,
    #                         test_size=0.15, random_state=5)

    X_train_path = 'final_data/X_train.npy'
    y_train_path = 'final_data/y_train.npy'
    X_val_path = 'final_data/X_val.npy'
    y_val_path = 'final_data/y_val.npy'
    X_test_path = 'final_data/X_test.npy'
    y_test_path = 'final_data/y_test.npy'

    check_class_balance(y_train_path, "Class distribution in y_train")
    check_class_balance(y_val_path, "Class distribution in y_val")
    check_class_balance(y_test_path,"Class distribution in y_test")

    plot_each_class_examples(X_train_path, y_train_path, 5)
    classifier = NeuralNetworkClassifier(X_train_path,y_train_path, X_val_path, y_val_path, X_test_path, y_test_path)
    # Train the model
    classifier.train(10,100)
    # Make predictions
    predictions = classifier.predict()
    # Evaluate model performance
    accuracy, precision, recall, f1, conf_matrix = classifier.evaluate(predictions)
    # Plot ROC curve
    classifier.plot_roc_curve(predictions)

    # Print additional evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Confusion Matrix:\n", conf_matrix)
