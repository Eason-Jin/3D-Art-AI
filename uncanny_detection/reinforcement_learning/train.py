from SAC_network import SACAgent
from environment import UncannyEnvironment
from replay_buffer import ReplayBuffer
import os
import cv2
import pandas as pd
import datetime
import numpy as np  # Import numpy

IMAGE_FOLDER = '../'
UNCANNY_FOLDER = os.path.join(IMAGE_FOLDER, 'uncanny')
NOT_UNCANNY_FOLDER = os.path.join(IMAGE_FOLDER, 'not_uncanny')


def load_images(folder, is_uncanny):
    images = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        print(f"Loading {filepath}")
        image = cv2.imread(filepath)
        if image is not None:
            images.append({'image': image, 'is_uncanny': is_uncanny})
    return images


uncanny_images = load_images(UNCANNY_FOLDER, True)
not_uncanny_images = load_images(NOT_UNCANNY_FOLDER, False)

env = UncannyEnvironment(pd.DataFrame(uncanny_images),
                         pd.DataFrame(not_uncanny_images))
STATE_DIM = 1  # accuracy
ACTION_DIM = 2  # confidence_threshold, low_conf_ratio_threshold
ACTION_RANGE = [0.1, 0.9]
INITIAL_THRESHOLDS = [0.4, 0.3]
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
    episode_reward = 0
    step = 0
    while True:
        if episode == 0 and step == 0:  # Use initial thresholds only for the first step
            action = agent.select_action(
                state, initial_thresholds=INITIAL_THRESHOLDS)
        else:
            action = agent.select_action(state)
        confidence_threshold, low_conf_ratio_threshold = action
        reward, next_state, done = env.step(
            confidence_threshold, low_conf_ratio_threshold)
        print(f"Episode {episode}, Step {step}")
        print(f"Action: {(confidence_threshold, low_conf_ratio_threshold)}, Reward: {reward}, Accuracy: {next_state[0]}")
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
        f.write(f"Accuracy: {state[0]}\n\n")

    print(f"Episode {episode}, Reward: {episode_reward}")
    print(f"Confidence Threshold: {confidence_threshold}")
    print(f"Low Confidence Ratio Threshold: {low_conf_ratio_threshold}")
    print(f"Accuracy: {state[0]}")
    print()
