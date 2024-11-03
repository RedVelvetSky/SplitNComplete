import keras
from keras.datasets import reuters
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

def create_testing_data():
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

    train = vectorize_sequences(train_data)
    test = vectorize_sequences(test_data)
    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)

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
    (train,train_labels,_,_) = input

    # specify the shape of the network
    model = models.Sequential()
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='silu', input_shape=(10000,)))
    model.add(layers.Dense(46, activation='softmax'))

    # split input data into training set and validation set
    val_data = train[:1000]
    train_data = train[1000:]

    val_labels = train_labels[:1000]
    train_labels = train_labels[1000:]

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
    input = create_testing_data()
    (train,train_labels,_,_) = input
    val_data = train[:1000]
    val_labels = train_labels[:1000]

    models_l = []

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