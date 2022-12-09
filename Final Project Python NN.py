import pandas
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.metrics
import sklearn.preprocessing
import keras

def train_and_evaluate(X_train, Y_train, X_test, Y_test):
    # Create layers (Functional API)
    # Input (2 dimensions)
    inputs = keras.layers.Input(shape=(11,), dtype='float32', name='input_layer')  
    
    # Hidden layer
    outputs = keras.layers.Dense(64, activation='relu', name='hidden_layer')(inputs)  
    
    # Output layer (3 labels)
    outputs = keras.layers.Dense(2, activation='softmax', name='output_layer')(outputs)  
    
    # Create a model from input layer and output layers
    model = keras.models.Model(inputs=inputs, outputs=outputs, name='neural_network')
    
    # Compile the model (binary_crossentropy if 2 classes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Convert labels to categorical: categorical_crossentropy expects targets
    # to be binary matrices (1s and 0s) of shape (samples, classes)
    Y_binary = keras.utils.to_categorical(Y_train, num_classes=2, dtype='int')
    
    # Train the model on the train set (output debug information)
    model.fit(X_train, Y_binary, batch_size=1, epochs=100, verbose=1)
    
    # Save the model (Make sure that the folder exists)
    model.save('./keras_nn.h5')
    
    # Evaluate on training data
    print('\n-- Training data --')
    predictions = model.predict(X_train)
    accuracy = sklearn.metrics.accuracy_score(Y_train, np.argmax(predictions, axis=1))
    print('Accuracy: {0:.2f}'.format(accuracy * 100.0))
    print('Classification Report:')
    print(sklearn.metrics.classification_report(Y_train, np.argmax(predictions, axis=1)))
    print('Confusion Matrix:')
    print(sklearn.metrics.confusion_matrix(Y_train, np.argmax(predictions, axis=1)))
    print('')
    
    # Evaluate on test data
    print('\n---- Test data ----')
    predictions = model.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(Y_test, np.argmax(predictions, axis=1))
    print('Accuracy: {0:.2f}'.format(accuracy * 100.0))
    print('Classification Report:')
    print(sklearn.metrics.classification_report(Y_test, np.argmax(predictions, axis=1)))
    print('Confusion Matrix:')
    print(sklearn.metrics.confusion_matrix(Y_test, np.argmax(predictions, axis=1)))

# The main entry point for this module
def main():
  
    #data = './MonkeyPox.csv'
    data = '~/desktop/MonkeyPox.csv'
    # define dictionaty to map TRUE / FLASE string to bool
    dic = {
        'Negative':0,
        'Positive':1,
        }

    dataset = pandas.read_csv(data)
    dataset = dataset.replace(dic)
    array = dataset.values

    # Slice data set in data and labels (2D-array)
    X = dataset.values[:, 0:11].astype(bool)  # Data
    Y = dataset.values[:, 11].astype(bool)  # Labels

    # Split data set in train and test (use random state to get the same split every time, and stratify to keep balance)
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=5,
                                                                                stratify=Y)
    # Make sure that data still is balanced
    print('\n--- Class balance ---')
    print(np.unique(Y_train, return_counts=True))
    print(np.unique(Y_test, return_counts=True))
    # Train and evaluate
    train_and_evaluate(X_train, Y_train, X_test, Y_test)

# Tell python to run main method
if __name__ == "__main__": main()
