import torch
import torch.optim as optim
import torch.nn as nn
from utils import DEVICE, normalize_reward, denormalize_reward


class SACAgent:
    def __init__(self, state_dim, action_dim, action_range):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic2 = Critic(state_dim, action_dim).to(DEVICE)
        self.target_critic1 = Critic(state_dim, action_dim).to(DEVICE)
        self.target_critic2 = Critic(state_dim, action_dim).to(DEVICE)

        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.action_range = action_range
        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state, initial_thresholds=None):
        if initial_thresholds is not None:
            # Use predefined initial thresholds
            action = initial_thresholds
            self.initialise_actor_with_thresholds(initial_thresholds)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action = self.actor(state).detach().cpu().numpy()[0]

            action = self.action_range[0] + (action + 1.0) * \
                0.5 * (self.action_range[1] - self.action_range[0])
        return action

    def initialise_actor_with_thresholds(self, initial_thresholds):
        # Initialize actor weights to output values close to initial thresholds
        with torch.no_grad():
            for param in self.actor.parameters():
                param.uniform_(-0.1, 0.1)  # Small random initialization
            # Adjust the final layer bias to approximate initial thresholds
            self.actor.fc3.bias.data = torch.FloatTensor(initial_thresholds).to(DEVICE)

    def update(self, replay_buffer, batch_size=64):
        # Sample a batch from the replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(
            batch_size)

        state = torch.FloatTensor(state).to(DEVICE)
        action = torch.FloatTensor(action).to(DEVICE)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(DEVICE)
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        done = torch.FloatTensor(done).unsqueeze(1).to(DEVICE)

        # # Normalize rewards
        # reward_mean = reward.mean()
        # reward_std = reward.std()
        # reward = normalize_reward(reward, reward_mean, reward_std)

        # Update Critic
        with torch.no_grad():
            next_action = self.actor(next_state)
            next_q1 = self.target_critic1(next_state, next_action)
            next_q2 = self.target_critic2(next_state, next_action)
            next_q = torch.min(next_q1, next_q2)
            target_q = reward + (1 - done) * self.gamma * next_q

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()

        # Gradient clipping for critic
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)

        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Update Actor
        actor_loss = -self.critic1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # Gradient clipping for actor
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)

        self.actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()

        # Initialize weights
        self._initialize_weights()

    def forward(self, state):
        state = state.to(DEVICE)
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        x = self.output_activation(self.fc3(x))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.fc(x)
