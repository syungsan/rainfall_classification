from tensorflow_docs.vis import embed
from tensorflow import keras
import pandas as pd
import numpy as np
import imageio
import cv2
import os
import codecs
import csv
import glob
from keras.models import load_model


IMG_SIZE = 224
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


def build_feature_extractor():
    _feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = _feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


def prepare_single_video(frames):
    feature_extractor = build_feature_extractor()

    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = read_csv("../data/labels.csv")[0]
    sequence_model = load_model("../save/model.h5")

    frames = load_video(path)
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames


# This utility is for visualization.
# Referenced from:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("../save/animation.gif", converted_images, duration=100)
    return embed.embed_file("../save/animation.gif")


class_names = os.listdir("../data/test")
test_datas = []

for class_name in class_names:
    _video_paths = glob.glob("../data/test/{}/*.avi".format(class_name))

    for video_path in _video_paths:
        test_datas.append([video_path, class_name])

test_df = pd.DataFrame(test_datas, columns=["video_path", "tag"])

test_video = np.random.choice(test_df["video_path"].values.tolist())
print(f"Test video path: {test_video}")
test_frames = sequence_prediction(test_video)
to_gif(test_frames[:MAX_SEQ_LENGTH])
