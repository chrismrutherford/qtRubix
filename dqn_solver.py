import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import time
import logging
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime

class DQN(nn.Module):
    def __init__(self, history_length=4, hidden_size=512, output_size=18):
        # Input: current state (54) + initial state (54) + last move (18) = 126
        input_size = 126
        super(DQN, self).__init__()
        #self.network = nn.Sequential(
        #    nn.Linear(input_size, 256),
        #    nn.ReLU(),
        #    nn.Linear(256, 128),
        #    nn.ReLU(),
        #    nn.Linear(128, output_size)
        #)
        self.network = nn.Sequential(
            nn.Linear(126, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        
    def forward(self, current_state, initial_state, last_move=None):
        # Create one-hot encoded move tensor if provided
        if last_move is None:
            move_tensor = torch.zeros(current_state.size(0), 18).to(current_state.device)
        else:
            move_tensor = torch.zeros(current_state.size(0), 18).to(current_state.device)
            move_tensor.scatter_(1, last_move.unsqueeze(1), 1)
            
        # Concatenate current state, initial state and move
        combined_state = torch.cat((current_state, initial_state, move_tensor), dim=1)
        return self.network(combined_state)

class RubiksCubeEnvironment:
    def __init__(self, cube, history_length=4):
        self.cube = cube
        self.history_length = history_length
        self.initial_state = None
        self.last_move = None
        self.last_move = None
        self.total_reward = 0
        self.step_count = 0
        self.action_space = [
            'F', 'B', 'U', 'D', 'L', 'R', 'M', 'E', 'S',
            "F'", "B'", "U'", "D'", "L'", "R'", "M'", "E'", "S'"
        ]
        self.color_map = {
            'white': 0, 'yellow': 1, 'red': 2,
            'orange': 3, 'blue': 4, 'green': 5
        }
        self.state_history = deque(maxlen=history_length)
        self.move_history = deque(maxlen=history_length)
        self.reward_history = deque(maxlen=history_length)
        
    def get_current_state(self):
        """Get the current cube state without history"""
        state = []
        for face in ['F', 'B', 'U', 'D', 'L', 'R']:
            for row in self.cube.faces[face]:
                for color in row:
                    state.append(self.color_map[color])
        return state

    def reset(self):
        """Reset the environment state"""
        self.total_reward = 0
        self.step_count = 0
        # Store current state as initial state
        self.initial_state = self.get_current_state()
        self.state_history.clear()
        self.move_history.clear()
        self.reward_history.clear()
        
    def get_state(self):
        """Get current state only"""
        return np.array(self.get_current_state(), dtype=np.float32)
    
    def step(self, action, ui_callback=None):
        # Calculate score before move
        previous_score = (self.cube.get_basic_score() + self.cube.get_advanced_score()) / 2
        
        # Store current state in history before making move
        self.state_history.append(self.get_current_state())
        
        # Store action and perform move
        self.last_move = action
        move = self.action_space[action]
        self.move_history.append(move)
        if ui_callback:
            ui_callback()
        if move[-1] == "'":  # Counterclockwise move
            if move[0] == 'U':
                self.cube.rotate_face_counterclockwise('D')
            elif move[0] == 'D':
                self.cube.rotate_face_counterclockwise('U')
            elif move[0] in ['F', 'B', 'L', 'R']:
                self.cube.rotate_face_counterclockwise(move[0])
            elif move[0] == 'M':
                self.cube.rotate_M_ccw()
            elif move[0] == 'E':
                self.cube.rotate_E_ccw()
            elif move[0] == 'S':
                self.cube.rotate_S_ccw()
        else:  # Clockwise move
            if move == 'U':
                self.cube.rotate_face_clockwise('D')
            elif move == 'D':
                self.cube.rotate_face_clockwise('U')
            elif move in ['F', 'B', 'L', 'R']:
                self.cube.rotate_face_clockwise(move)
            elif move == 'M':
                self.cube.rotate_M()
            elif move == 'E':
                self.cube.rotate_E()
            elif move == 'S':
                self.cube.rotate_S()
        
        # Get new state
        new_state = self.get_state()
        
        # Calculate new score and reward
        new_score = (self.cube.get_basic_score() + self.cube.get_advanced_score()) / 2
        
        # Calculate reward based on score improvement
        score_change = new_score - previous_score
        
        reward = score_change / 50.0  # Increased from 100 to make rewards more pronounced

        if score_change < 0:
            reward *= 1.5  # Reduced penalty from 2.0
        elif score_change == 0:
            reward = -0.05  # Reduced penalty for neutral moves

        done = new_score == 100

        if done:
            reward = 20.0  # Increased solve reward from 10.0
                
        # Clear history if episode is done
        if done:
            self.state_history.clear()
            self.move_history.clear()
            self.reward_history.clear()
        
        return new_state, reward, done

class RubiksCubeSolver:
    def __init__(self, cube):
        # Set up logging
        self.setup_logging()
        
        # Initialize success/failure tracking
        self.total_attempts = 0
        self.successful_solves = 0
        self.failed_solves = 0
        self.success_rates = []  # Track success rate over time
        self.plot_interval = 100  # Update plot every 100 attempts
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            
        self.env = RubiksCubeEnvironment(cube)
        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        
        # Log network architecture details
        self.logger.info("\n=== Neural Network Architecture ===")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Network structure:\n{self.model}")
        self.logger.info("===============================\n")
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.solved_count = 0
        self.total_attempts = 0
        
        # Initialize target model with current model weights first
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Then try to load existing model if it exists
        self.load_model('rubiks_model.pth')
        self.memory = deque(maxlen=10000)  # Store 10000 states in memory
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.target_update = 10
        self.training = False
        
    def save_model(self, filename='rubiks_model.pth'):
        """Save the current model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        
    def load_model(self, filename='rubiks_model.pth'):
        """Load a saved model state"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
        
    def remember(self, state, action, reward, next_state, done, cube_state_str, move=None):
        """Store a complete memory entry with state, action, reward, and move information"""
        memory_entry = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'cube_state': cube_state_str,
            'move': move,  # Store the actual move made
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        }
        self.memory.append(memory_entry)
        
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 17)
        
        current_state = torch.FloatTensor(state).view(1, -1).to(self.device)
        initial_state = torch.FloatTensor(self.env.initial_state).view(1, -1).to(self.device)
        last_move = torch.LongTensor([self.env.last_move]).to(self.device) if self.env.last_move is not None else None
        with torch.no_grad():
            q_values = self.model(current_state, initial_state, last_move)
        return q_values.argmax().item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        if self.total_attempts % 1000 == 0:  # Print debug info every 1000 attempts
            print(f"\nReplay Buffer Size: {len(self.memory)}")
            print(f"Current Epsilon: {self.epsilon:.3f}")
        
        # Skip replay if initial state is not set
        if self.env.initial_state is None:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert numpy arrays to tensors and move to device
        current_states = torch.FloatTensor(np.vstack([entry['state'] for entry in batch])).to(self.device)
        initial_states = torch.FloatTensor(np.vstack([self.env.initial_state for _ in batch])).to(self.device)
        actions = torch.LongTensor([entry['action'] for entry in batch]).to(self.device)
        rewards = torch.FloatTensor([entry['reward'] for entry in batch]).to(self.device)
        next_states = torch.FloatTensor(np.vstack([entry['next_state'] for entry in batch])).to(self.device)
        dones = torch.FloatTensor([entry['done'] for entry in batch]).to(self.device)
        
        # States and next_states are already tensors on device from earlier vstack
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        current_q_values = self.model(current_states, initial_states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states, initial_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def setup_logging(self):
        """Initialize logging configuration"""
        self.logger = logging.getLogger('RubiksSolver')
        self.last_log_time = 0  # Track last log time
        
        # Only set up handlers if they haven't been added yet
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            
            # Create logs directory if it doesn't exist
            if not os.path.exists('logs'):
                os.makedirs('logs')
                
            # File handler for detailed logging
            fh = logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            fh.setLevel(logging.INFO)
            
            # Console handler for basic output
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            # Add handlers
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
    
    def train(self, callback=None):
        """Single training attempt using specified solve steps"""
        self.training = True
        total_reward = 0
        scramble_moves = []
        solution_moves = []
        show_logs = (self.total_attempts % 10000 == 0)  # Reduced frequency
        
        # Reset environment and cube
        self.env.reset()
        self.env.cube.reset()
        for _ in range(self.env.cube.scramble_steps):
            action = random.randint(0, 17)
            self.env.step(action)
            scramble_moves.append(self.env.action_space[action])
        
        # Store initial state after scramble
        scrambled_state = self.get_cube_state_str()
        
        # Try to solve
        state = self.env.get_state()
        for step in range(self.env.cube.solve_steps):
            action = self.act(state)
            next_state, reward, done = self.env.step(action, callback)
            total_reward += reward
            solution_moves.append(self.env.action_space[action])
            
            current_state_str = self.get_cube_state_str()
            self.remember(state, action, reward, next_state, done, current_state_str, 
                         move=self.env.action_space[action])
            self.replay()
            
            if done:
                # Calculate final score
                final_score = (self.env.cube.get_basic_score() + self.env.cube.get_advanced_score()) / 2
                
                # Log successful solve with both states
                self.successful_solves += 1
                success_rate = (self.successful_solves / self.total_attempts) * 100
                self.success_rates.append(success_rate)
                
                current_time = time.time()
                if current_time - self.last_log_time >= 1.0:  # Only log if 1 second has passed
                    self.logger.info("\n=== Successful Solve ===")
                    self.logger.info(f"Scramble Moves: {', '.join(scramble_moves)}")
                    self.logger.info(f"Solution Moves: {', '.join(solution_moves)}")
                    self.logger.info(f"Total Moves: {len(scramble_moves) + len(solution_moves)}")
                    self.logger.info(f"Final Score: {final_score:.1f}%")
                    self.logger.info(f"Success/Fail: {self.successful_solves}/{self.failed_solves}")
                    self.logger.info(f"Success Rate: {success_rate:.2f}% ({self.successful_solves}/{self.total_attempts})")
                    self.logger.info(f"Initial Scrambled State:\n{scrambled_state}")
                    self.logger.info(f"Final Solved State:\n{self.get_cube_state_str()}")
                    self.logger.info("==================\n")
                    self.last_log_time = current_time
                
                self.solved_count += 1
                break
            
            state = next_state
        
        self.total_attempts += 1
        if not done:
            self.failed_solves += 1
            
        # Calculate final reward based on cube score
        final_score = (self.env.cube.get_basic_score() + self.env.cube.get_advanced_score()) / 2
        total_reward = final_score / 100  # Linear scaling from 0-100% to 0-1
        
        # Update plot periodically
        if self.total_attempts % self.plot_interval == 0:
            self.plot_success_rate()
        
        if show_logs:  # Only show periodic status updates
            success_rate = (self.successful_solves / self.total_attempts) * 100
            current_time = time.time()
            if current_time - self.last_log_time >= 1.0:  # Only log if 1 second has passed
                self.logger.info("\n=== Status Update ===")
                self.logger.info(f"Total Attempts: {self.total_attempts}")
                self.logger.info(f"Solved: {self.successful_solves}/{self.total_attempts} ({success_rate:.2f}%)")
                self.logger.info(f"Current Epsilon: {self.epsilon:.3f}")
                self.logger.info("==================\n")
                self.last_log_time = current_time
            
        if total_reward > 0 and callback:
            callback()  # Trigger success animation
        
        self.training = False
        
    def plot_success_rate(self):
        """Plot the success rate over time"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.success_rates)), self.success_rates, 'b-', label='Success Rate')
        plt.xlabel('Training Iterations (x100)')
        plt.ylabel('Success Rate (%)')
        plt.title('Rubik\'s Cube Solver Success Rate Over Time')
        plt.grid(True)
        plt.legend()
        
        # Save plot
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig('plots/success_rate.png')
        plt.close()

    def get_cube_state_str(self):
        """Convert current cube state to string representation"""
        state_str = []
        for face in ['F', 'B', 'U', 'D', 'L', 'R']:
            colors = [color[0].upper() for row in self.env.cube.faces[face] for color in row]
            state_str.append(f"{face}:[{''.join(colors)}]")
        return ' '.join(state_str)

    def solve(self, max_steps=1000000):
        """Attempt to solve the cube using the trained model"""
        if self.training:
            return
            
        # Get initial scrambled state
        scrambled_state = self.get_cube_state_str()
        state = self.env.get_state()
        total_reward = 0
        solution = []
        moves_made = []  # Initialize moves_made list
        state_history = [(scrambled_state, None)]  # (state, move) pairs
        
        for step in range(max_steps):
            action = self.act(state)
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            move = self.env.action_space[action]
            solution.append(move)
            current_state = self.get_cube_state_str()
            state_history.append((current_state, move))
            
            if done:
                print(f"Solved in {step + 1} moves!")
                # Log the successful solution
                self.logger.info("\n=== Successful Solve ===")
                self.logger.info(f"Scramble Moves: {', '.join(moves_made[:self.env.cube.scramble_steps])}")
                self.logger.info(f"Solution Moves: {', '.join(solution)}")
                self.logger.info(f"Total Moves: {len(solution)}")
                self.logger.info(f"Initial Scrambled State:\n{scrambled_state}")
                self.logger.info(f"Final Solved State:\n{state_history[-1][0]}")
                self.logger.info("\nMove History:")
                for i, (state, move) in enumerate(state_history):
                    if i == 0:
                        self.logger.info(f"Initial state:\n{state}")
                    elif move:  # Only show states after actual moves
                        self.logger.info(f"\nAfter move {move}:\n{state}")
                break
                
            state = next_state
        
        return solution, total_reward, scrambled_state, state_history
