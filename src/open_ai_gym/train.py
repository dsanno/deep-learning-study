import argparse
import gym
import six
import numpy as np
import random

import chainer
from chainer import functions as F
from chainer import links as L

class Agent(chainer.Chain):
    gamma = 0.99
    initial_epsilon = 1
    epsilon_reduction = 0.001
    min_epsilon = 0.01

    def __init__(self, input_size, output_size, hidden_size):
        initialW = chainer.initializers.HeNormal(0.01)
        super(Agent, self).__init__(
            fc1=F.Linear(input_size, hidden_size, initialW=initialW),
            fc2=F.Linear(hidden_size, hidden_size, initialW=initialW),
            fc3=F.Linear(hidden_size, output_size, initialW=initialW),
        )
        self.epsilon = self.initial_epsilon
        self.output_size = output_size

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h

    def randomize_action(self, action):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_size - 1)
        return action

    def reduce_epsilon(self):
        self.epsilon = (self.epsilon - self.min_epsilon) * (1 - self.epsilon_reduction) + self.min_epsilon

    def adjust_reward(self, state, reward, done):
        return reward

    def normalize_state(self, state):
        return np.asarray(state, dtype=np.float32)

class CartPoleAgent(Agent):
    gamma = 0.9
    initial_epsilon = 1
    min_epsilon = 0.01
    epsilon_reduction = 0.001

    def __init__(self):
        super(CartPoleAgent, self).__init__(4, 2, 24)

    def adjust_reward(self, state, reward, done):
        return reward

    def normalize_state(self, state):
        scale = np.asarray([1 / 2.4, 1 / 4.0, 1 / 0.2, 1 / 3.0], dtype=np.float32)
        return np.asarray(state, dtype=np.float32) * scale

class MountainCarAgent(Agent):
    gamma = 0.99
    initial_epsilon = 0.8
    min_epsilon = 0.1
    epsilon_reduction = 0.0001

    def __init__(self):
        super(MountainCarAgent, self).__init__(2, 3, 64)

    def adjust_reward(self, state, reward, done):
        return reward

    def normalize_state(self, state):
        scale = np.asarray([1 / 1.2, 1 / 0.07], dtype=np.float32)
        return np.asarray(state, dtype=np.float32) * scale

class ExperiencePool(object):

    def __init__(self, size, state_shape):
        self.size = size
        self.states = np.zeros(((size,) + state_shape), dtype=np.float32)
        self.actions = np.zeros((size,), dtype=np.int32)
        self.rewards = np.zeros((size,), dtype=np.float32)
        self.nexts = np.zeros((size,), dtype=np.float32)
        self.pos = 0

    def add(self, state, action, reward, done):
        index = self.pos % self.size
        self.states[index, ...] = state
        self.actions[index] = action
        self.rewards[index] = reward
        if done:
            self.nexts[index] = 0
        else:
            self.nexts[index] = 1
        self.pos += 1

    def available_size(self):
        if self.pos > self.size:
            return self.size - 1
        return self.pos - 1

    def __getitem__(self, index):
        if self.pos < self.size:
            offset = 0
        else:
            offset = self.pos % self.size - self.size
        index += offset
        return self.states[index], self.actions[index], self.rewards[index], self.states[index + 1], self.nexts[index]

def update(agent, target_agent, optimizer, ex_pool, batch_size):
    available_size = ex_pool.available_size()
    if available_size < batch_size:
        return
    indices = np.random.permutation(available_size)[:batch_size]
    data = [ex_pool[i] for i in indices]
    state, action, reward, next_state, has_next = zip(*data)
    state = np.asarray(state)
    action = np.asarray(action)
    reward = np.asarray(reward)
    next_state = np.asarray(next_state)
    has_next = np.asarray(has_next)

    q = F.select_item(agent(state), action)
    next_action = np.argmax(agent(next_state).data, axis=1)
    y = reward + agent.gamma * has_next * target_agent(next_state).data[(six.moves.range(len(next_action))), next_action]
    loss = F.mean_squared_error(q, y)
    agent.cleargrads()
    loss.backward()
    optimizer.update()

def parse_arg():
    parser = argparse.ArgumentParser('Open AI Gym learning sample')
    parser.add_argument('--env', '-e', type=str, choices=['cart_pole', 'mountain_car'], help='Environment name')
    parser.add_argument('--skip_render', '-s', type=int, default=0, help='Episodes nterval to skip rendering')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size for taining')
    parser.add_argument('--pool-size', '-p', type=int, default=2000, help='Experiance pool size')
    parser.add_argument('--train-iter', '-t', type=int, default=10, help='Number of training iterations')
    parser.add_argument('--episode', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--episode-len', type=int, default=1000, help='Length of an episode')
    parser.add_argument('--use-double-q', action='store_true', help='Use Double Q-learning')
    return parser.parse_args()

def main():
    args = parse_arg()
    episode_num = args.episode
    episode_length = args.episode_len
    pool_size = args.pool_size
    batch_size = args.batch_size
    train_num = args.train_iter
    update_count = 0
    update_agent_interval = 100
    use_double_q = args.use_double_q

    env_name = args.env
    if env_name == 'mountain_car':
        env = gym.make('MountainCar-v0')
        agent = MountainCarAgent()
    else:
        env = gym.make('CartPole-v0')
        agent = CartPoleAgent()
    skip_rendering_interval = args.skip_render

    if use_double_q:
        target_agent = agent.copy()
    else:
        target_agent = agent
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(agent)
    ex_pool = ExperiencePool(pool_size, env.observation_space.shape)

    for episode in six.moves.range(episode_num):
        raw_state = env.reset()
        state = agent.normalize_state(raw_state)
        need_render = skip_rendering_interval <= 0 or episode % skip_rendering_interval == 0
        for t in six.moves.range(episode_length):
            if need_render:
                env.render()
            action = np.argmax(agent(np.expand_dims(state, 0)).data)
            action = agent.randomize_action(action)

            prev_state = state
            raw_state, raw_reward, done, info = env.step(action)
            reward = agent.adjust_reward(raw_state, raw_reward, done)
            state = agent.normalize_state(raw_state)
            ex_pool.add(prev_state, action, reward, done or t == episode_length - 1)
            for i in six.moves.range(train_num):
                update(agent, target_agent, optimizer, ex_pool, batch_size)
            update_count += 1
            agent.reduce_epsilon()
            if use_double_q and update_count % update_agent_interval == 0:
                target_agent = agent.copy()
            if done:
                print('Episode {} finished after {} timesteps'.format(episode + 1, t + 1))
                break
        if not done:
            print('Epsode {} completed'.format(episode + 1))

if __name__ == '__main__':
    main()
