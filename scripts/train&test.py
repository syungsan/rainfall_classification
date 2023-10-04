from tensorflow import keras
import numpy as np
import codecs
import csv
import matplotlib.pyplot as plt
import os
import shutil


BATCH_SIZE = 64
EPOCHS = 30

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048


def read_csv(file_path, delimiter=","):

    array = []
    file = codecs.open(file_path, "r", "utf-8")

    reader = csv.reader(file, delimiter=delimiter)

    for line in reader:
        array.append(line)

    file.close()
    return array


def write_csv(file_path, array):

    try:
        # 書き込み UTF-8
        with open(file_path, "w", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            writer.writerows(array)

    # 起こりそうな例外をキャッチ
    except FileNotFoundError as e:
        print(e)
    except csv.Error as e:
        print(e)


# Utility for our sequence model.
def get_sequence_model():
    class_vocab = read_csv("../data/labels.csv")[0]
    print(class_vocab)

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model


def plot_result(history):

    """
    plot result
    全ての学習が終了した後に、historyを参照して、accuracyとlossをそれぞれplotする
    """

    # accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="acc", marker=".")
    plt.plot(history.history["val_accuracy"], label="val_acc", marker=".")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.grid()
    plt.legend(loc="best")
    plt.title("Accuracy")
    plt.savefig("../save/graphs/accuracy.png")
    plt.show()

    # loss
    plt.figure()
    plt.plot(history.history["loss"], label="loss", marker=".")
    plt.plot(history.history["val_loss"], label="val_loss", marker=".")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.grid()
    plt.legend(loc="best")
    plt.title("Loss")
    plt.savefig("../save/graphs/loss.png")
    plt.show()


# Utility for running experiments.
def run_experiment():

    train_features = np.array(read_csv("../data/train_features.csv"), dtype=float)
    train_masks = np.array(read_csv("../data/train_masks.csv"), dtype=bool)
    train_labels = np.array(read_csv("../data/train_labels.csv"), dtype=int)

    test_features = np.array(read_csv("../data/test_features.csv"), dtype=float)
    test_masks = np.array(read_csv("../data/test_masks.csv"), dtype=bool)
    test_labels = np.array(read_csv("../data/test_labels.csv"), dtype=int)

    train_features = np.reshape(train_features, (train_features.shape[0], MAX_SEQ_LENGTH, NUM_FEATURES))
    test_features = np.reshape(test_features, (test_features.shape[0], MAX_SEQ_LENGTH, NUM_FEATURES))

    print(train_features.shape)
    print(test_features.shape)

    filepath = "../save/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_features, train_masks],
        train_labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    loss, accuracy = seq_model.evaluate([test_features, test_masks], test_labels)

    print(f"Test loss: {loss}")
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    write_csv("../save/result.csv", [["loss", loss], ["accuracy", accuracy * 100]])

    return history, seq_model


history, sequence_model = run_experiment()

if os.path.isdir("../save/graphs"):
    shutil.rmtree("../save/graphs")
os.mkdir("../save/graphs")

plot_result(history)

sequence_model.save("../save/model.h5")
