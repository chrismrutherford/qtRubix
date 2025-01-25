import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

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

class RubiksSolver:
    def __init__(self):
        # State size: 6 faces * 9 stickers * 6 possible colors (one-hot)
        self.state_size = 6 * 9 * 6
        # Action size: 12 possible moves (F,B,U,D,L,R) * (clockwise/counterclockwise)
        self.action_size = 12
        self.memory = deque(maxlen=20000)  # Larger memory
        self.gamma = 0.95  # Reduced discount factor for shorter-term rewards
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Lower minimum exploration
        self.epsilon_decay = 0.997  # Adjusted decay rate
        self.learning_rate = 0.001  # Standard learning rate
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(self.state_size, self.action_size).to(self.device)
        self.target_model = DQN(self.state_size, self.action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.update_target_every = 1000  # Update target network every N steps
        self.steps = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def get_state(self, cube):
        # Convert cube state to one-hot encoded tensor
        state = []
        # Map face positions to standard colors
        face_color_map = {
            'U': 'white', 'R': 'red', 'F': 'green',
            'D': 'yellow', 'L': 'orange', 'B': 'blue'
        }
        color_map = {'white': 0, 'red': 1, 'green': 2, 
                    'yellow': 3, 'orange': 4, 'blue': 5}
        
        for face in ['U', 'R', 'F', 'D', 'L', 'B']:
            for row in cube.faces[face]:
                for color in row:
                    # One-hot encode each color
                    one_hot = [0] * 6
                    # Map the color through both mappings
                    color_idx = color_map[color]
                    one_hot[color_idx] = 1
                    state.extend(one_hot)
                    
        return torch.FloatTensor(state).to(self.device)
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state = state.unsqueeze(0)
            q_values = self.model(state)
            return q_values.argmax().item()
            
    def remember(self, state, action, reward, next_state, done):
        # Validate experiences before storing
        if torch.isnan(state).any() or torch.isnan(next_state).any():
            print("Warning: NaN detected in state")
            return
            
        if abs(reward) > 10:
            print(f"Warning: Large reward detected: {reward}")
            
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = torch.stack([x[0] for x in batch])
        actions = torch.tensor([x[1] for x in batch], device=self.device)
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([x[3] for x in batch])
        dones = torch.tensor([x[4] for x in batch], dtype=torch.float32, device=self.device)
        
        # Current Q-values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: Use online network for action selection and target network for value estimation
        with torch.no_grad():
            best_actions = self.model(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_model(next_states).gather(1, best_actions).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Huber loss for better stability
        loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
