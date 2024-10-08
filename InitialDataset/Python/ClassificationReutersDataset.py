import keras
from keras.datasets import reuters
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

def create_testing_data():
    """
    Data = a newswire represented as a sequence of integers (representing words) 
    Labels = one of 46 categories the newswire talks about
    Consider only the 10000 most often used words, vectorization produces a vector of length 10000, index i = 1 if word i is in the newswire
    One hot labels is a vector of length 46 with 1 at the position of the correct category
    """
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

    train = vectorize_sequences(train_data)
    test = vectorize_sequences(test_data)
    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)

    # For fun, we can decode the input data to see what a newswire looks like
    # decode_input_data(train_data)

    return (train, one_hot_train_labels, test, one_hot_test_labels)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def decode_input_data(train_data):
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    # Note that our indices were offset by 3
    # because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
    decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
    print(decoded_newswire)

def create_and_train_network(input, index):
    """
    Create a network with the input of size 10000, two hidden layers, and one output layer of size 46
    The output of the network is a vector of probabilities the newswire falls into the specific category
    Set aside 1000 samples for validation, use the rest for training
    """
    (train,train_labels,_,_) = input

    # task 2 and 3
    '''
    Using significantly fewer neurons than the output size in a neural network may lead to a bottleneck effect, limiting the network's capacity to capture complex patterns and reducing its ability to learn from the data effectively, potentially resulting in underfitting and poor performance. It can also increase the risk of information loss and hinder the model's ability to represent the input data's full complexity, impacting its predictive power.
    '''
    # specify the shape of the network
    model = models.Sequential()
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='silu', input_shape=(10000,)))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(256, activation='silu'))
    model.add(layers.Dense(46, activation='softmax'))
    '''    
    Milestones (все изменения выполнены путем Grid Search)
    1. Поменяли кол-во нейронов с 64 до 128, слоев с 2 до 1. Поменяли оптимайзер на Адам. Поменяли количество эпох и batch_size
    2. Добавили Learning Rate Schedule (CosineDecay) и установили шагизменения в len(train) * epoch_num, поменяли initial Learning Rate
    3. Добавили дропаут 0.2
    4. Поменяли функцию активации hidden layers на silu
    5. Добавили label_smoothing=0.1
    6. Добавили сallback для сохранения модели которая лучше всего себя показала на val set
    7. Создали ансамбль из трех моделей
    8. Финальная точность примерно 87%, было 82.4%

    '''   

    # split input data into training set and validation set
    val_data = train[:1000]
    train_data = train[1000:]

    val_labels = train_labels[:1000]
    train_labels = train_labels[1000:]

    # task 4
    '''
    The disparity in accuracy and loss between the training and validation sets typically suggests that the model is overfitting. When accuracy on the training set is high while accuracy on the validation set lags behind, it indicates that the model is memorizing the training data rather than generalizing well to unseen data. This observation implies that the model may need to be trained for fewer epochs or that regularization techniques should be employed to prevent overfitting and improve generalization performance.
    '''
    cos_dec = keras.optimizers.schedules.CosineDecay(
        0.0001,
        len(train) * 25,
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cos_dec), 
                loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                metrics=['accuracy'])
    
    checkpoint_filepath = f'checkpoint_mode_{index + 1}.keras'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # train the network
    history = model.fit(train_data,
                        train_labels,
                        epochs=30,
                        batch_size=32,
                        validation_data=(val_data, val_labels),
                        callbacks=[model_checkpoint_callback])
    
    return (history,model)



def print_graphs(history):
    """
    History contains data about the training process. It contains an entry for each metric used for both training and validation.
    Specifically, we plot loss = difference between the expected outcome and the produced outcome
    and accuracy = fraction of predictions the model got right
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()   # clear figure

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


if __name__ == "__main__":

    # prepare data (task 1)
    '''
    The model's performance on the testing set is more comparable to its performance on the validation set, as both serve to estimate its generalization ability to unseen data, while the training set is used for parameter optimization.
    '''
    input = create_testing_data()
    (train,train_labels,_,_) = input
    val_data = train[:1000]
    val_labels = train_labels[:1000]

    # натренировали модель и сохранили лучшие веса
    models_l = []
    # for i in range(3):
    #     # create and train the neural network
    #     (history,model) = create_and_train_network(input, i)
    #      # show the results
    #     print_graphs(history)

    for i in range(3):
        model = keras.models.load_model(f'checkpoint_mode_{i + 1}.keras')
        model.fit(train, train_labels)
        models_l.append(model)

    ensemble_input = keras.layers.Input(shape=(10000,))
    outputs = [model(ensemble_input) for model in models_l]
    ensemble_output = keras.layers.Average()(outputs)

    ensemble = keras.Model(inputs=ensemble_input, outputs=ensemble_output)
    ensemble.compile(loss=keras.losses.CategoricalCrossentropy(),
                     metrics=['accuracy'])
    
    print("Dev accuracy: ")
    acc = ensemble.evaluate(val_data, val_labels)