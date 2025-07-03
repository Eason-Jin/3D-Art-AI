import pickle
import pandas as pd
from ultralytics import YOLO


class Environment:
    def __init__(self, data: pd.DataFrame):
        self.data = data  # Image, True/False
        self.current_index = 0
        self.model = YOLO('yolo11x.pt')

    def reset(self):
        self.current_index = 0
        self.data = self.data.sample(frac=1).reset_index(
            drop=True)
        
        return 0

    def step(self, action):
        confidence_threshold, low_conf_ratio_threshold = action

        # Calculate accuracy over the entire dataset
        correct_count = 0
        for index in range(len(self.data)):
            detection = self.get_detection(index)
            is_uncanny = self.is_image_uncanny(detection)
            if is_uncanny == self.data[index][1]:
                correct_count += 1

        accuracy = correct_count / len(self.data)

        # Reward based on the current image
        detection = self.get_detection(self.current_index)
        is_uncanny = self.is_image_uncanny(detection)
        reward = 1 if is_uncanny == self.data[self.current_index][1] else -1

        self.current_index += 1
        done = self.current_index >= len(self.data)
        if done:
            print(
                f"\nConfidence Threshold: {confidence_threshold}, Low Confidence Ratio Threshold: {low_conf_ratio_threshold}")
            with open('results.pkl', 'wb') as f:
                pickle.dump({
                    'confidence_threshold': confidence_threshold,
                    'low_conf_ratio_threshold': low_conf_ratio_threshold,
                }, f)

        return reward, accuracy, done

    def get_detection(self, index):
        self.model(self.data[index][0])[0]

    def is_image_uncanny(self, detection):
        if len(detection.boxes) == 0:
            return True
        else:
            low_conf_count = sum(
                float(box.conf[0]) < self.confidence_threshold for box in detection.boxes)
            low_conf_ratio = low_conf_count / len(detection.boxes)
            return low_conf_ratio > self.low_conf_ratio_threshold