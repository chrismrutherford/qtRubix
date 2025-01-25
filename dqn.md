Deep Q-Network (DQN) for Solving Rubik's Cube

This document provides a detailed explanation of the Deep Q-Network (DQN) implementation for solving a Rubik's Cube. The code combines concepts from reinforcement learning (RL), deep learning, and PyQt5 for visualization..
Table of Contents

    Introduction to DQN

    Key Components of the DQN

        Neural Network Architecture

        Experience Replay

        Epsilon-Greedy Strategy

        Target Network

        Reward Calculation

    Rubik's Cube Environment

        State Representation

        Action Space

        Reward Function

    Training Process

        Exploration vs. Exploitation

        Experience Replay and Training

        Target Network Updates

    Visualization and Interaction

        PyQt5 Interface

        Cube Visualization

    Conclusion

1. Introduction to DQN

A Deep Q-Network (DQN) is a reinforcement learning algorithm that combines Q-learning with deep neural networks. It is used to solve problems where the state space is large or continuous, making traditional Q-learning infeasible. The key idea is to use a neural network to approximate the Q-value function, which estimates the expected cumulative reward for taking an action in a given state.

In this implementation, the DQN is trained to solve a Rubik's Cube by learning a policy that maps cube states to actions (moves) that maximize the cumulative reward.
2. Key Components of the DQN
2.1 Neural Network Architecture

The neural network used in the DQN is defined in the DQN class:
python
Copy

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

    Input Size: The input size corresponds to the state representation of the Rubik's Cube. Each face of the cube has 9 stickers, and each sticker can be one of 6 colors. The state is one-hot encoded, resulting in an input size of 6 * 9 * 6 = 324.

    Output Size: The output size corresponds to the number of possible actions. There are 12 possible moves (6 faces × 2 directions: clockwise and counterclockwise).

    Layers:

        Two hidden layers with 256 and 128 neurons, respectively.

        Layer normalization (nn.LayerNorm) is used to stabilize training.

        Dropout (nn.Dropout) is applied to prevent overfitting.

        ReLU activation functions introduce non-linearity.

2.2 Experience Replay

Experience replay is a technique used to break the correlation between consecutive experiences. The agent stores experiences (state, action, reward, next state, done) in a replay buffer and samples mini-batches from it during training.
python
Copy

self.memory = deque(maxlen=20000)  # Replay buffer with a maximum size of 20,000 experiences

    Replay Buffer: A deque (double-ended queue) is used to store experiences. When the buffer is full, the oldest experiences are removed to make space for new ones.

    Mini-Batch Sampling: During training, a mini-batch of experiences is randomly sampled from the replay buffer to update the network.

2.3 Epsilon-Greedy Strategy

The epsilon-greedy strategy balances exploration and exploitation. Initially, the agent explores the environment by taking random actions. As training progresses, the agent increasingly exploits the learned policy.
python
Copy

self.epsilon = 1.0  # Initial exploration rate
self.epsilon_min = 0.05  # Minimum exploration rate
self.epsilon_decay = 0.997  # Decay rate for epsilon

    Epsilon Decay: After each episode, epsilon is decayed by multiplying it with epsilon_decay. This reduces the exploration rate over time.

    Random Action: With probability epsilon, the agent takes a random action.

    Greedy Action: With probability 1 - epsilon, the agent takes the action with the highest Q-value.

2.4 Target Network

A target network is used to stabilize training. The target network is a copy of the main network, but its weights are updated less frequently.
python
Copy

self.target_model = DQN(self.state_size, self.action_size).to(self.device)
self.target_model.load_state_dict(self.model.state_dict())
self.update_target_every = 1000  # Update target network every 1,000 steps

    Target Q-Values: The target network is used to compute the target Q-values during training. This reduces the risk of divergence caused by using the same network to predict both the current and target Q-values.

    Periodic Updates: The target network is updated every update_target_every steps by copying the weights from the main network.

2.5 Reward Calculation

The reward function is designed to encourage the agent to solve the cube efficiently. The reward is based on the improvement in the cube's completion score.
python
Copy

def calculate_reward(self, old_score, new_score):
    reward = (new_score - old_score) / 20.0  # Reward for improvement
    if new_score == 100:  # Bonus for solving the cube
        reward += 5.0
    elif new_score < old_score:  # Penalty for worsening the state
        reward -= 0.1
    reward -= 0.01  # Small penalty for each move to encourage efficiency
    return reward

    Improvement Reward: The agent receives a reward proportional to the improvement in the cube's completion score.

    Solving Bonus: A large bonus is given if the cube is solved.

    Penalty: A small penalty is applied if the cube's state worsens or for each move taken.

3. Rubik's Cube Environment
3.1 State Representation

The state of the Rubik's Cube is represented as a one-hot encoded tensor. Each face of the cube has 9 stickers, and each sticker can be one of 6 colors. The state is flattened into a 1D tensor of size 6 * 9 * 6 = 324.
python
Copy

def get_state(self, cube):
    state = []
    for face in ['U', 'R', 'F', 'D', 'L', 'B']:
        for row in cube.faces[face]:
            for color in row:
                one_hot = [0] * 6
                color_idx = color_map[color]
                one_hot[color_idx] = 1
                state.extend(one_hot)
    return torch.FloatTensor(state).to(self.device)

3.2 Action Space

The action space consists of 12 possible moves:

    6 faces: Front (F), Back (B), Up (U), Down (D), Left (L), Right (R).

    2 directions: Clockwise (1) and Counterclockwise (3).

python
Copy

self.action_size = 12

3.3 Reward Function

The reward function is designed to guide the agent toward solving the cube. It rewards improvements in the cube's completion score and penalizes inefficient moves.
4. Training Process
4.1 Exploration vs. Exploitation

The agent uses the epsilon-greedy strategy to balance exploration and exploitation. Initially, the agent explores the environment by taking random actions. As training progresses, the agent increasingly exploits the learned policy.
4.2 Experience Replay and Training

During training, the agent stores experiences in the replay buffer and samples mini-batches to update the network. The loss function is the Huber loss, which is less sensitive to outliers than the mean squared error.
python
Copy

loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)

4.3 Target Network Updates

The target network is updated periodically to stabilize training. The weights of the target network are copied from the main network every update_target_every steps.
5. Visualization and Interaction

The PyQt5 interface allows users to interact with the Rubik's Cube and visualize the training process. The cube is displayed in both 2D and 3D, and users can perform moves, scramble the cube, and train the DQN.
6. Conclusion

This implementation demonstrates how a DQN can be used to solve a Rubik's Cube. By combining reinforcement learning with deep neural networks, the agent learns to map cube states to actions that maximize the cumulative reward. The use of experience replay, target networks, and the epsilon-greedy strategy ensures stable and efficient training.

This document provides a comprehensive overview of the DQN and its application to the Rubik's Cube problem. Students familiar with ANNs should now have a solid understanding of how DQNs work and how they can be applied to complex problems.
