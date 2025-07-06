import pandas as pd
from ultralytics import YOLO
from utils import DEVICE, INITIAL_THRESHOLDS


class UncannyEnvironment:
    def __init__(self, uncanny_data: pd.DataFrame, not_uncanny_data: pd.DataFrame):
        self.train_data, self.test_data = self.split_data(
            uncanny_data, not_uncanny_data)
        self.current_index = 0
        self.model = YOLO('yolo11x.pt', verbose=False).to(DEVICE)

    def reset(self):
        self.current_index = 0
        accuracy, precision, recall = self.calculate_metrics(
            INITIAL_THRESHOLDS[0], INITIAL_THRESHOLDS[1])
        self.prev_accuracy = accuracy - 0.1
        self.prev_precision = precision - 0.1
        self.prev_recall = recall - 0.1
        return INITIAL_THRESHOLDS + [self.prev_accuracy, self.prev_precision, self.prev_recall, self.current_index]

    def step(self, confidence_threshold, low_conf_ratio_threshold):
        # Calculate effect of current actions
        accuracy, precision, recall = self.calculate_metrics(
            confidence_threshold, low_conf_ratio_threshold)
        print(
            f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

        # Classify on the current image
        detection = self.get_detection(self.current_index)
        is_uncanny = self.is_image_uncanny(
            detection, confidence_threshold, low_conf_ratio_threshold)
        is_correct = is_uncanny == self.train_data.iloc[self.current_index, 1]
        reward = self.calculate_reward(
            is_correct, accuracy, precision, recall)

        self.prev_accuracy = accuracy
        self.prev_precision = precision
        self.prev_recall = recall
        self.current_index += 1
        done = self.current_index >= len(self.train_data)

        return reward, [confidence_threshold, low_conf_ratio_threshold, accuracy, precision, recall, self.current_index/len(self.train_data)], done

    """
      _____      _            _         ______                _   _
     |  __ \    (_)          | |       |  ____|              | | (_)
     | |__) | __ ___   ____ _| |_ ___  | |__ _   _ _ __   ___| |_ _  ___  _ __  ___
     |  ___/ '__| \ \ / / _` | __/ _ \ |  __| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
     | |   | |  | |\ V / (_| | ||  __/ | |  | |_| | | | | (__| |_| | (_) | | | \__ \
     |_|   |_|  |_| \_/ \__,_|\__\___| |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
    
    """

    def split_data(self, uncanny_data, not_uncanny_data):
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

    def get_detection(self, index):
        image = self.train_data.iloc[index, 0]
        return self.model(image, verbose=False)[0]

    def calculate_metrics(self, confidence_threshold, low_conf_ratio_threshold):
        correct_count = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for i in range(len(self.test_data)):
            detection = self.get_detection(i)
            is_uncanny = self.is_image_uncanny(
                detection, confidence_threshold, low_conf_ratio_threshold)
            actual_label = self.test_data.iloc[i, 1]

            if is_uncanny == actual_label:
                correct_count += 1
                if is_uncanny:  # True positive
                    true_positive += 1
            else:
                if is_uncanny:  # False positive
                    false_positive += 1
                else:  # False negative
                    false_negative += 1

        accuracy = float(correct_count / len(self.test_data)) * 100
        precision = (true_positive / (true_positive + false_positive)) * \
            100 if (true_positive + false_positive) > 0 else 0
        recall = (true_positive / (true_positive + false_negative)) * \
            100 if (true_positive + false_negative) > 0 else 0

        return accuracy, precision, recall

    def is_image_uncanny(self, detection, confidence_threshold, low_conf_ratio_threshold):
        if len(detection.boxes) == 0:
            return True
        else:
            low_conf_count = sum(
                float(box.conf[0]) < confidence_threshold for box in detection.boxes)
            low_conf_ratio = low_conf_count / len(detection.boxes)
            return low_conf_ratio > low_conf_ratio_threshold

    def calculate_reward(self, is_correct, accuracy, precision, recall):
        r1, r2, r3 = 0.2, 0.8, 0.5

        current_reward = 10 if is_correct else -10

        accuracy_reward = self.calculate_metric_reward(
            accuracy, self.prev_accuracy)
        precision_reward = self.calculate_metric_reward(
            precision, self.prev_precision)
        recall_reward = self.calculate_metric_reward(
            recall, self.prev_recall)

        reward = r1 * current_reward + \
            r2 * (accuracy + precision + recall) + \
            r3 * (accuracy_reward + precision_reward + recall_reward)

        return reward

    def calculate_metric_reward(self, val, prev_val):
        delta = val - prev_val
        return -10 if delta <= 0 else delta
