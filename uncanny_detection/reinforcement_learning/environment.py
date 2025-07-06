import pandas as pd
from ultralytics import YOLO
from utils import DEVICE


class UncannyEnvironment:
    def __init__(self, uncanny_data: pd.DataFrame, not_uncanny_data: pd.DataFrame):
        self.train_data, self.test_data = self.__split_data(
            uncanny_data, not_uncanny_data)
        self.current_index = 0
        self.model = YOLO('yolo11x.pt', verbose=False).to(DEVICE)
        self.prev_accuracy = 0.0

    def __split_data(self, uncanny_data, not_uncanny_data):
        uncanny_count = len(uncanny_data)
        not_uncanny_count = len(not_uncanny_data)

        min_size = min(uncanny_count, not_uncanny_count)
        if uncanny_count != not_uncanny_count:
            print(
                f"Unbalanced dataset! Uncanny: {uncanny_count}, Not Uncanny: {not_uncanny_count}")
            print("Removing images to balance dataset")
            uncanny_data = uncanny_data.sample(
                min_size).reset_index(drop=True)
            not_uncanny_data = not_uncanny_data.sample(
                min_size).reset_index(drop=True)

        uncanny_shuffled = uncanny_data.sample(
            frac=1).reset_index(drop=True)
        not_uncanny_shuffled = not_uncanny_data.sample(
            frac=1).reset_index(drop=True)

        split_idx = int(0.7 * min_size)

        uncanny_train = uncanny_shuffled.iloc[:split_idx]
        uncanny_test = uncanny_shuffled.iloc[split_idx:]

        not_uncanny_train = not_uncanny_shuffled.iloc[:split_idx]
        not_uncanny_test = not_uncanny_shuffled.iloc[split_idx:]

        train_data = pd.concat([uncanny_train, not_uncanny_train]).sample(
            frac=1).reset_index(drop=True)
        test_data = pd.concat([uncanny_test, not_uncanny_test]).sample(
            frac=1).reset_index(drop=True)
        return train_data, test_data

    def reset(self):
        self.current_index = 0
        self.prev_accuracy = 0.0
        return [0.0]  # Initial accuracy is 0

    def step(self, confidence_threshold, low_conf_ratio_threshold):
        # Calculate accuracy over the test dataset
        correct_count = 0
        for i in range(len(self.test_data)):
            detection = self.get_detection(i)
            is_uncanny = self.is_image_uncanny(
                detection, confidence_threshold, low_conf_ratio_threshold)
            if is_uncanny == self.test_data.iloc[i, 1]:
                correct_count += 1

        accuracy = correct_count / len(self.test_data)

        # Classify on the current image
        detection = self.get_detection(self.current_index)
        is_uncanny = self.is_image_uncanny(
            detection, confidence_threshold, low_conf_ratio_threshold)
        is_correct = is_uncanny == self.train_data.iloc[self.current_index, 1]
        reward = self.calculate_reward(
            is_correct, accuracy, self.prev_accuracy)

        self.prev_accuracy = accuracy
        self.current_index += 1
        done = self.current_index >= len(self.train_data)

        return reward, [accuracy], done

    def get_detection(self, index):
        image = self.train_data.iloc[index, 0]
        return self.model(image, verbose=False)[0]

    def is_image_uncanny(self, detection, confidence_threshold, low_conf_ratio_threshold):
        if len(detection.boxes) == 0:
            return True
        else:
            low_conf_count = sum(
                float(box.conf[0]) < confidence_threshold for box in detection.boxes)
            low_conf_ratio = low_conf_count / len(detection.boxes)
            return low_conf_ratio > low_conf_ratio_threshold

    def calculate_reward(self, is_correct, accuracy, prev_accuracy):
        current_reward = 1 if is_correct else -1
        accuracy_delta = accuracy - prev_accuracy
        return 0.2 * current_reward + 0.8 * accuracy_delta
