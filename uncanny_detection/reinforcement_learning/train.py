from SAC_network import SACAgent
from environment import UncannyEnvironment
from replay_buffer import ReplayBuffer
import os
import pandas as pd
import datetime

from utils import INITIAL_THRESHOLDS, load_images, UNCANNY_FOLDER, NOT_UNCANNY_FOLDER


uncanny_images = load_images(UNCANNY_FOLDER, True)
not_uncanny_images = load_images(NOT_UNCANNY_FOLDER, False)

env = UncannyEnvironment(pd.DataFrame(uncanny_images),
                         pd.DataFrame(not_uncanny_images))
# confidence_threshold, low_conf_ratio_threshold, accuracy, precision, recall, current_index/len(train_data)
STATE_DIM = 6
ACTION_DIM = 2  # confidence_threshold, low_conf_ratio_threshold
ACTION_RANGE = [0.01, 0.99]
agent = SACAgent(STATE_DIM, ACTION_DIM, ACTION_RANGE)
replay_buffer = ReplayBuffer(max_size=100000)

num_episodes = 100
batch_size = 64

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

reward_dir = os.path.join('uncanny_classification', timestamp, 'rewards')
if not os.path.exists(reward_dir):
    os.makedirs(reward_dir)
with open(f'{reward_dir}/rewards.csv', 'w') as f:
    f.write("Episode,Reward,Conf,LowConf,Accuracy,Precision,Recall\n")

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
        confidence_threshold = max(ACTION_RANGE[0], min(
            confidence_threshold, ACTION_RANGE[1]))
        low_conf_ratio_threshold = max(ACTION_RANGE[0], min(
            low_conf_ratio_threshold, ACTION_RANGE[1]))
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

    with open(f'{reward_dir}/rewards.csv', 'a') as f:
        f.write(
            f"{episode},{episode_reward},{confidence_threshold},{low_conf_ratio_threshold},{state[2]},{state[3]},{state[4]}\n")
