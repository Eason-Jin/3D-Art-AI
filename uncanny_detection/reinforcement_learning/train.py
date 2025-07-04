# Initialize environment, agent, and replay buffer
from SAC_network import SACAgent
from environment import UncannyEnvironment
from replay_buffer import ReplayBuffer
import os
import cv2
import pandas as pd


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

env = UncannyEnvironment(pd.DataFrame(uncanny_images), pd.DataFrame(not_uncanny_images))
STATE_DIM = 1  # accuracy
ACTION_DIM = 2  # confidence_threshold, low_conf_ratio_threshold
ACTION_RANGE = [0, 1]
INITIAL_THRESHOLDS = [0.4, 0.3]
agent = SACAgent(STATE_DIM, ACTION_DIM, ACTION_RANGE)
replay_buffer = ReplayBuffer(max_size=100000)

num_episodes = 1000
batch_size = 64

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    while True:
        print(f"State: {state}")
        if episode < 1:
            action = agent.select_action(
                state, initial_thresholds=INITIAL_THRESHOLDS)
        else:
            action = agent.select_action(state)
        confidence_threshold, low_conf_ratio_threshold = action
        reward, next_state, done = env.step(confidence_threshold, low_conf_ratio_threshold)
        print(f"Next State: {next_state}")
        replay_buffer.add(state, action, reward, next_state, done)

        if len(replay_buffer.buffer) > batch_size:
            agent.update(replay_buffer, batch_size)

        state = next_state
        episode_reward += reward

        if done:
            break

    print(f"Episode {episode + 1}, Reward: {episode_reward}")
