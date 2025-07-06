from SAC_network import SACAgent
from environment import UncannyEnvironment
from replay_buffer import ReplayBuffer
import os
import cv2
import pandas as pd
import datetime
from imgaug import augmenters as iaa
from utils import INITIAL_THRESHOLDS

IMAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
UNCANNY_FOLDER = os.path.join(IMAGE_FOLDER, 'uncanny')
NOT_UNCANNY_FOLDER = os.path.join(IMAGE_FOLDER, 'not_uncanny')


def load_images(folder, is_uncanny):
    images = []

    # Define individual augmenters
    flip_augmenter = iaa.Fliplr(1.0)  # Flip all images horizontally
    contrast_augmenter = iaa.LinearContrast((0.75, 1.5))  # Adjust contrast

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        print(f"Loading {filepath}")
        image = cv2.imread(filepath)
        if image is not None:
            # Original image
            images.append({'image': image, 'is_uncanny': is_uncanny})

            # Flipped image
            flipped_image = flip_augmenter(image=image)
            images.append({'image': flipped_image, 'is_uncanny': is_uncanny})

            # Contrast-adjusted image
            contrast_image = contrast_augmenter(image=image)
            images.append({'image': contrast_image, 'is_uncanny': is_uncanny})

            # Both flipped and contrast-adjusted image
            flipped_contrast_image = contrast_augmenter(image=flipped_image)
            images.append({'image': flipped_contrast_image,
                          'is_uncanny': is_uncanny})

    return images


uncanny_images = load_images(UNCANNY_FOLDER, True)
not_uncanny_images = load_images(NOT_UNCANNY_FOLDER, False)

env = UncannyEnvironment(pd.DataFrame(uncanny_images),
                         pd.DataFrame(not_uncanny_images))
# confidence_threshold, low_conf_ratio_threshold, accuracy, precision, recall, current_index/len(train_data)
STATE_DIM = 6
ACTION_DIM = 2  # confidence_threshold, low_conf_ratio_threshold
ACTION_RANGE = [0.1, 0.9]
agent = SACAgent(STATE_DIM, ACTION_DIM, ACTION_RANGE)
replay_buffer = ReplayBuffer(max_size=100000)

num_episodes = 10
batch_size = 64

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
reward_dir = os.path.join('uncanny_classification', timestamp, 'rewards')
if not os.path.exists(reward_dir):
    os.makedirs(reward_dir)

for episode in range(num_episodes):
    state = env.reset()
    assert len(
        state) == STATE_DIM, f"State should have {STATE_DIM} dimensions, got {len(state)}"
    episode_reward = 0
    step = 0
    while True:
        print(f"Episode {episode}, Step {step}")
        if episode == 0 and step == 0:  # Use initial thresholds only for the first step
            action = agent.select_action(
                state, initial_thresholds=INITIAL_THRESHOLDS)
        else:
            action = agent.select_action(state)
        confidence_threshold, low_conf_ratio_threshold = action
        reward, next_state, done = env.step(
            confidence_threshold, low_conf_ratio_threshold)
        assert len(
            next_state) == STATE_DIM, f"Next state should have {STATE_DIM} dimensions, got {len(next_state)}"
        print(
            f"Action: {(confidence_threshold, low_conf_ratio_threshold)}, Reward: {reward}")
        print()
        replay_buffer.add(state, action, reward, next_state, done)

        if len(replay_buffer.buffer) > batch_size:
            agent.update(replay_buffer, batch_size)

        state = next_state
        episode_reward += reward
        step += 1
        if done:
            break

    with open(f'{reward_dir}/rewards.txt', 'a') as f:
        f.write(f"Episode {episode}, Reward: {episode_reward}\n")
        f.write(f"Confidence Threshold: {confidence_threshold}\n")
        f.write(
            f"Low Confidence Ratio Threshold: {low_conf_ratio_threshold}\n")
        f.write(f"Accuracy: {state[2]}\n\n")
