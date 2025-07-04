import torch
import torch.optim as optim
import torch.nn as nn
from utils import DEVICE


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
        self.alpha = 0.2  # Entropy coefficient

    def select_action(self, state, initial_thresholds=None):
        if initial_thresholds is not None:
            # Use predefined initial thresholds
            return initial_thresholds

        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        action = self.actor(state).detach().numpy()[0]
        # Scale action to the desired range
        action = self.action_range[0] + (action + 1.0) * \
            0.5 * (self.action_range[1] - self.action_range[0])
        return action

    def update(self, replay_buffer, batch_size=64):
        # Sample a batch from the replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(
            batch_size)

        state = torch.FloatTensor(state).to(DEVICE)
        action = torch.FloatTensor(action).to(DEVICE)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(DEVICE)
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        done = torch.FloatTensor(done).unsqueeze(1).to(DEVICE)

        # Update Critic
        with torch.no_grad():
            next_action = self.actor(next_state)
            next_q1 = self.target_critic1(next_state, next_action)
            next_q2 = self.target_critic2(next_state, next_action)
            next_q = torch.min(next_q1, next_q2) - \
                self.alpha * torch.log(next_action)
            target_q = reward + (1 - done) * self.gamma * next_q

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Update Actor
        actor_loss = -self.critic1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
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
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Outputs actions in range [-1, 1]
        )

    def forward(self, state):
        return self.fc(state)


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
