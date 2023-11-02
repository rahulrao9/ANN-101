import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def input_check(data):
    if not isinstance(data, (list, np.ndarray)):
        raise ValueError("The input must be a list or NumPy ndarray")


# Split the data into training and testing sets
# input: 1) x: list/ndarray (features)
#        2) y: list/ndarray (target)
# output: split: tuple of X_train, X_test, y_train, y_test
def split_and_standardize(X, y):
    # lets consider a 80-20 split with random state as NONE
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    scalar = StandardScaler()
    x_train = scalar.fit_transform(x_train)
    x_test = scalar.transform(x_test)

    return x_train, x_test, y_train, y_test


# input:  1) X_train: list/ndarray
#         2) y_train: list/ndarray

# output: 1) models: model1,model2 - tuple
def create_model(X_train, y_train):
    model1 = MLPClassifier(hidden_layer_sizes=(25, 50, 25), activation='relu', max_iter=100, random_state=1)
    model1.fit(X_train, y_train)

    # Create and train the second MLP classifier with different parameters
    model2 = MLPClassifier(hidden_layer_sizes=(2, 4, 2), activation='identity', max_iter=100, random_state=42)
    model2.fit(X_train, y_train)

    return model1, model2


# input  : 1) model: MLPClassifier after training
#          2) X_train: list/ndarray
#          3) y_train: list/ndarray
# output : 1) metrics: tuple - accuracy,precision,recall,fscore,confusion matrix
def predict_and_evaluate(model, X_test, y_test):
    # TODO
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='micro')
    recall = recall_score(y_test, predictions, average='micro')
    f1score = f1_score(y_test, predictions, average='micro')
    conf_mat = confusion_matrix(y_test, predictions)
    return accuracy, precision, recall, f1score, conf_mat
