import gym
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import random
from collections import deque
import matplotlib.pyplot as plt


class DuelingQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DuelingQNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, 1)
        self.fc_advantage = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def len(self):
        return len(self.buffer)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*transitions)
        return np.array(obs), actions, rewards, np.array(next_obs), dones

    def clean(self):
        self.buffer.clear()


class DQN:
    def __init__(self, env, input_size, hidden_size, output_size):
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_net = DuelingQNet(input_size, hidden_size, output_size).to(self.device)
        self.target_net = DuelingQNet(input_size, hidden_size, output_size).to(self.device)
        self.optim = optim.Adam(self.eval_net.parameters(), lr=args.lr)
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self.buffer = ReplayBuffer(args.capacity)
        self.loss_fn = nn.MSELoss()
        self.learn_step = 0
        self.steps_done = 0

    def choose_action(self, obs):
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.steps_done / self.eps_decay)
        self.steps_done += 1
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        obs = torch.Tensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            actions = self.eval_net(obs)
        return torch.argmax(actions).item()

    def store_transition(self, *transition):
        self.buffer.push(*transition)

    def learn(self):
        if self.learn_step % args.update_target == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1

        if self.buffer.len() < args.batch_size:
            return

        obs, actions, rewards, next_obs, dones = self.buffer.sample(args.batch_size)
        obs = torch.Tensor(obs).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_obs = torch.Tensor(next_obs).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_eval = self.eval_net(obs).gather(1, actions)
        q_next = self.target_net(next_obs).max(1)[0].detach().unsqueeze(1)
        q_target = rewards + args.gamma * q_next * (1 - dones)

        loss = self.loss_fn(q_eval, q_target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


def main():
    env = gym.make(args.env)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    agent = DQN(env, o_dim, args.hidden, a_dim)
    rewards = []
    avg_rewards = []

    for i_episode in range(args.n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step_cnt = 0
        while not done and step_cnt < 500:
            step_cnt += 1
            env.render()
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, done)
            episode_reward += reward
            obs = next_obs
            if agent.buffer.len() >= args.batch_size:
                agent.learn()

        episode_reward = int(episode_reward)
        rewards.append(episode_reward)
        average_reward = int(np.mean(rewards[-100:]))
        avg_rewards.append(average_reward)
        print(f"Episode: {i_episode + 1}, Reward: {episode_reward}, Average Reward Per 100 Episodes: {average_reward}")

    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Over Episodes')

    plt.figure()
    plt.plot(avg_rewards)
    plt.ylim(0, 500)
    plt.yticks(range(0, 501, 25))
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Per 100 Episodes')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="environment name")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--hidden", default=64, type=int, help="dimension of hidden layer")
    parser.add_argument("--n_episodes", default=500, type=int, help="number of episodes")
    parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
    parser.add_argument("--capacity", default=10000, type=int, help="capacity of replay buffer")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--update_target", default=100, type=int, help="frequency to update target network")
    parser.add_argument("--eps_start", default=0.9, type=float)
    parser.add_argument("--eps_end", default=0.01, type=float)
    parser.add_argument("--eps_decay", default=500, type=int)
    args = parser.parse_args()
    main()
