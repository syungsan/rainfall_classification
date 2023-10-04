from tensorflow import keras
import pandas as pd
import numpy as np
import cv2
import os
import glob
import csv


IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048


train_datas = []
test_datas = []
datas = [train_datas, test_datas]

for index, data_path in enumerate(["../data/train", "../data/test"]):
    class_names = os.listdir(data_path)

    for class_name in class_names:
        _video_paths = glob.glob("{}/{}/*.avi".format(data_path, class_name))

        for video_path in _video_paths:
            datas[index].append([video_path, class_name])

train_df = pd.DataFrame(train_datas, columns=["video_path", "tag"])
test_df = pd.DataFrame(test_datas, columns=["video_path", "tag"])

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")
print(train_df.sample(10))


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


def flatten_with_any_depth(nested_list):

    """深さ優先探索の要領で入れ子のリストをフラットにする関数"""
    # フラットなリストとフリンジを用意
    flat_list = []
    fringe = [nested_list]

    while len(fringe) > 0:
        node = fringe.pop(0)

        # ノードがリストであれば子要素をフリンジに追加
        # リストでなければそのままフラットリストに追加
        if isinstance(node, list):
            fringe = node + fringe
        else:
            flat_list.append(node)

    return flat_list


# The following two methods are taken from this tutorial:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub


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


feature_extractor = build_feature_extractor()

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
)
print(label_processor.get_vocabulary())

write_csv("../data/labels.csv", [label_processor.get_vocabulary()])


def prepare_all_videos(df):
    num_samples = len(df)
    video_paths = df["video_path"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH * NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(path)
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx, ] = np.array(flatten_with_any_depth(temp_frame_features.squeeze().tolist()))
        frame_masks[idx, ] = temp_frame_mask.squeeze()

        print("{}/{} processed...".format(idx + 1, len(video_paths)))

    return (frame_features, frame_masks), labels


train_data, train_labels = prepare_all_videos(train_df)

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")

write_csv("../data/train_features.csv", train_data[0])
write_csv("../data/train_masks.csv", train_data[1])
write_csv("../data/train_labels.csv", train_labels)

test_data, test_labels = prepare_all_videos(test_df)

write_csv("../data/test_features.csv", test_data[0])
write_csv("../data/test_masks.csv", test_data[1])
write_csv("../data/test_labels.csv", test_labels)

print("\nAll process was completed...")
