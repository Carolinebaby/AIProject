import gym
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import random
from collections import deque
import matplotlib.pyplot as plt


class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        # 使用双端队列存储经验，设置最大容量
        self.buffer = deque(maxlen=capacity)

    def len(self):
        return len(self.buffer)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        # 随机采样一批经验
        transitions = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*transitions)
        return np.array(obs), actions, rewards, np.array(next_obs), dones

    def clean(self):
        self.buffer.clear()


# Deep Q-Learning Network
class DQN:
    def __init__(self, env, input_size, hidden_size, output_size):
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 评估网络
        self.eval_net = QNet(input_size, hidden_size, output_size).to(self.device)
        # 目标网络
        self.target_net = QNet(input_size, hidden_size, output_size).to(self.device)
        # Adam优化器
        self.optim = optim.Adam(self.eval_net.parameters(), lr=args.lr)
        # ε-贪心策略的参数 ε 随着游戏回合的增加，逐渐减少
        # 在训练开始时，智能体更多地进行探索（即有较高的概率选择随机动作）
        # 在训练快要结束时，智能体更多地进行利用（即有较低的概率选择随机动作）
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        # ε 会在约 eps_decay 个步骤内从 eps_start 线性衰减到接近 eps_end
        self.eps_decay = args.eps_decay
        # 创建经验回放缓冲区
        self.buffer = ReplayBuffer(args.capacity)
        self.loss_fn = nn.MSELoss()
        self.learn_step = 0
        self.steps_done = 0

    # 选择动作 - 使用ε-贪心策略
    def choose_action(self, obs):
        # ε 的概率进行探索（随机选择一个动作）
        # 1-ε 的概率选择最佳动作
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.steps_done / self.eps_decay)
        self.steps_done += 1
        if np.random.rand() < epsilon:
            # 生成0到1之间的随机数，如果小于ε，则进行探索，选择随机动作
            return self.env.action_space.sample()
        # 1-ε 的概率： 根据当前策略选择最优动作
        obs = torch.Tensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():  # 关闭梯度计算
            actions = self.eval_net(obs)
        return torch.argmax(actions).item()  # 返回 Q 值最大的动作

    def store_transition(self, *transition):
        self.buffer.push(*transition)

    def learn(self):
        if self.learn_step % args.update_target == 0:  # 定期更新目标网络
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1

        # 从缓冲区采样一个批次的经验
        obs, actions, rewards, next_obs, dones = self.buffer.sample(args.batch_size)
        # 转换为 pytorch 张量
        obs = torch.Tensor(obs).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_obs = torch.Tensor(next_obs).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 计算评估网络的Q值
        q_eval = self.eval_net(obs).gather(1, actions)
        # 计算目标网络的Q值
        q_next = self.target_net(next_obs).max(1)[0].detach().unsqueeze(1)
        q_target = rewards + args.gamma * q_next * (1 - dones)

        # 计算损失
        loss = self.loss_fn(q_eval, q_target)

        # 反向传播和优化
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


def main():
    # 创建环境
    env = gym.make(args.env)
    # 获取环境的状态和动作空间维度
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    # 创建DQN智能体
    agent = DQN(env, o_dim, args.hidden, a_dim)
    rewards = []      # 存储每个游戏回合的奖励
    avg_rewards = []  # 存储每 100 个回合的平均奖励

    # 进行训练
    for i_episode in range(args.n_episodes):
        obs = env.reset()     # 游戏重置
        episode_reward = 0    # 奖励
        done = False          # 游戏是否结束
        step_cnt = 0          # 执行步骤的次数
        while not done and step_cnt < 500:
            step_cnt += 1
            env.render()      # 游戏画面渲染
            action = agent.choose_action(obs)  # 选择动作并执行
            next_obs, reward, done, info = env.step(action)  # 执行动作
            # 存储经验
            agent.store_transition(obs, action, reward, next_obs, done)
            episode_reward += reward  # 游戏回合的奖励
            obs = next_obs   # 更新状态
            # 如果缓冲区中的经验数量足够，则进行学习
            if agent.buffer.len() >= args.batch_size:
                agent.learn()

        episode_reward = int(episode_reward)
        rewards.append(episode_reward)
        average_reward = int(np.mean(rewards[-100:]))
        avg_rewards.append(average_reward)
        print(f"Episode: {i_episode + 1}, Reward: {episode_reward}, Average Reward Per 100 Episodes: {average_reward}")

    # 绘制奖励曲线
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
