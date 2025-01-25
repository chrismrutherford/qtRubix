from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QPushButton, QGraphicsView, QGraphicsScene, QApplication,
                            QLabel, QOpenGLWidget, QMessageBox, QSpinBox, QProgressDialog)
from dqnSolve import RubiksSolver
import torch
import random
from twophase import solver as sv, face as fc, cubie
from twophase.enums import Color, Move
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtCore import Qt, QTimer
from OpenGL.GL import *
from OpenGL.GLU import *
import sys
import math

class RubiksCube:
    def __init__(self):
        # Initialize faces according to Color enum order:
        # U=0 (white), R=1 (red), F=2 (green), D=3 (yellow), L=4 (orange), B=5 (blue)
        self.default_faces = {
            'U': [['white']*3 for _ in range(3)],   # U is white (0)
            'R': [['red']*3 for _ in range(3)],     # R is red (1)
            'F': [['green']*3 for _ in range(3)],   # F is green (2)
            'D': [['yellow']*3 for _ in range(3)],  # D is yellow (3)
            'L': [['orange']*3 for _ in range(3)],  # L is orange (4)
            'B': [['blue']*3 for _ in range(3)]     # B is blue (5)
        }
        self.reset()
        
    def reset(self):
        # Reset initial state
        self.initial_state = None
        # Initialize cube faces: front, back, up, down, left, right
        # Each face is a 3x3 grid represented by colors
        self.colors = {
            'white': QColor('white'),
            'yellow': QColor('yellow'),
            'red': QColor('red'),
            'orange': QColor('orange'),
            'blue': QColor('blue'),
            'green': QColor('green')
        }
        
        # Initialize faces with their default colors
        self.faces = {face: [row[:] for row in face_data] 
                     for face, face_data in self.default_faces.items()}

    def get_basic_score(self):
        """Calculate completion score based on number of moves to solve
        0 moves = 100%, more moves = lower score"""
        # Convert current state to facelet string
        facelet_str = ''
        for f in ['U', 'R', 'F', 'D', 'L', 'B']:
            for row in self.faces[f]:
                for col in row:
                    # Map colors to facelet letters
                    if col == 'white': facelet_str += 'U'
                    elif col == 'red': facelet_str += 'R' 
                    elif col == 'green': facelet_str += 'F'
                    elif col == 'yellow': facelet_str += 'D'
                    elif col == 'orange': facelet_str += 'L'
                    elif col == 'blue': facelet_str += 'B'
        
        # Get solution from twophase solver
        solution = sv.solve(facelet_str, 20, 2)  # max 20 moves, 2 sec timeout
        
        # Extract number of moves from solution string
        num_moves = int(solution.split('(')[1].split('f')[0])
        
        # Convert to percentage - 0 moves = 100%, 20 moves = 0%
        score = max(0, 100 - (num_moves * 5))
        return score
        

    def apply_move(self, face, clockwise=True):
        """Apply a move to the cube faces using twophase solver's CubieCube"""
        # Convert face name to Move enum value
        move_name = face + ("1" if clockwise else "3")
        move = getattr(Move, move_name)
        
        # Convert current state to CubieCube
        facecube = fc.FaceCube()
        # Build facelet string in URFDLB order
        facelet_str = ''
        for f in ['U', 'R', 'F', 'D', 'L', 'B']:
            for row in self.faces[f]:
                for col in row:
                    # Map colors to facelet letters
                    if col == 'white': facelet_str += 'U'
                    elif col == 'red': facelet_str += 'R'
                    elif col == 'green': facelet_str += 'F'
                    elif col == 'yellow': facelet_str += 'D'
                    elif col == 'orange': facelet_str += 'L'
                    elif col == 'blue': facelet_str += 'B'
        
        facecube.from_string(facelet_str)
        cc = facecube.to_cubie_cube()
        
        # Apply the move using twophase's move tables
        cc.multiply(cubie.moveCube[move])
        
        # Convert back to face colors
        facecube = cc.to_facelet_cube()
        facelet_str = facecube.to_string()
        
        # Update the faces dictionary
        idx = 0
        for f in ['U', 'R', 'F', 'D', 'L', 'B']:
            for i in range(3):
                for j in range(3):
                    # Map facelet letters back to colors
                    if facelet_str[idx] == 'U': self.faces[f][i][j] = 'white'
                    elif facelet_str[idx] == 'R': self.faces[f][i][j] = 'red'
                    elif facelet_str[idx] == 'F': self.faces[f][i][j] = 'green'
                    elif facelet_str[idx] == 'D': self.faces[f][i][j] = 'yellow'
                    elif facelet_str[idx] == 'L': self.faces[f][i][j] = 'orange'
                    elif facelet_str[idx] == 'B': self.faces[f][i][j] = 'blue'
                    idx += 1





class CubeFaceView(QWidget):
    def __init__(self, face_name):
        super().__init__()
        self.face_name = face_name
        
        layout = QVBoxLayout(self)
        layout.setSpacing(2)  # Small gap between view and label
        layout.setContentsMargins(0, 0, 0, 2)  # Small bottom margin
        
        # Add graphics view
        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.view.setFixedSize(120, 120)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.view)
        
        # Add label below view
        self.label = QLabel(face_name)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            QLabel { 
                color: white; 
                font-weight: bold;
                font-size: 12px;
                background-color: #444;
                padding: 0px 1px;
                border-radius: 2px;
                height: 12px;
            }
        """)
        layout.addWidget(self.label)

    def update_face(self, face_data, colors):
        self.scene.clear()
        # Draw cube squares
        for i in range(3):
            for j in range(3):
                rect = self.scene.addRect(i*30, j*30, 28, 28)
                color = colors[face_data[j][i]]
                rect.setBrush(QBrush(color))

class CubeGLWidget(QOpenGLWidget):
    def __init__(self, parent=None, cube=None):
        super().__init__(parent)
        self.rotation_x = 30
        self.rotation_y = -45
        self.last_pos = None
        self.cube = cube

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w/h, 0.1, 100.0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, -10)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        
        # Draw cube faces
        if self.cube:
            self.draw_face('F')  # Front
            self.draw_face('B', 180, 0, 1, 0)  # Back
            self.draw_face('U', -90, 1, 0, 0)  # Up
            self.draw_face('D', 90, 1, 0, 0)  # Down
            self.draw_face('L', -90, 0, 1, 0)  # Left
            self.draw_face('R', 90, 0, 1, 0)  # Right

    def draw_face(self, face_name, rot_angle=0, rot_x=0, rot_y=0, rot_z=0):
        if not self.cube:
            return
            
        glPushMatrix()
        glRotatef(rot_angle, rot_x, rot_y, rot_z)
        glTranslatef(0, 0, 1.5)
        
        glBegin(GL_QUADS)
        for i in range(3):
            for j in range(3):
                x1, y1 = -1.4 + i*0.9, -1.4 + j*0.9
                x2, y2 = x1 + 0.8, y1 + 0.8
                color_name = self.cube.faces[face_name][j][i]
                color = self.cube.colors[color_name]
                glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, 
                           [color.redF(), color.greenF(), color.blueF(), 1.0])
                glVertex3f(x1, y1, 0)
                glVertex3f(x2, y1, 0)
                glVertex3f(x2, y2, 0)
                glVertex3f(x1, y2, 0)
        glEnd()
        glPopMatrix()

    def mousePressEvent(self, event):
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        self.rotation_y += dx
        self.rotation_x += dy
        self.last_pos = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.last_pos = None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cube = RubiksCube()
        self.move_history = []  # Track moves made
        self.solver = RubiksSolver()
        # Load existing network if available
        try:
            self.solver.load_model('rubiks_dqn.pth')
            print("Loaded existing DQN model:")
            print(self.solver.model)
        except:
            print("Starting with new DQN model:")
            print(self.solver.model)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Rubik's Cube Simulator")
        self.setGeometry(100, 100, 600, 400)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Add 3D view
        self.gl_widget = CubeGLWidget(cube=self.cube)
        self.gl_widget.setFixedHeight(300)
        layout.addWidget(self.gl_widget)

        # Create 2D cube views in Facelet enum layout
        faces_layout = QGridLayout()
        faces_layout.setSpacing(2)
        self.views = {}
        
        # Create all face views first
        for face in ['U', 'R', 'F', 'D', 'L', 'B']:
            view = CubeFaceView(face)
            self.views[face] = view
            
        # Arrange in Facelet enum layout
        # Add U face (top)
        faces_layout.addWidget(self.views['U'], 0, 1)
        
        # Add middle row (L, F, R, B)
        faces_layout.addWidget(self.views['L'], 1, 0)
        faces_layout.addWidget(self.views['F'], 1, 1)
        faces_layout.addWidget(self.views['R'], 1, 2)
        faces_layout.addWidget(self.views['B'], 1, 3)
        
        # Add D face (bottom)
        faces_layout.addWidget(self.views['D'], 2, 1)
        
        layout.addLayout(faces_layout)

        # Create move buttons
        moves_layout = QVBoxLayout()
        
        # Clockwise moves (1)
        cw_moves = ['F1', 'B1', 'U1', 'D1', 'L1', 'R1'] 
        cw_layout = QHBoxLayout()
        for move in cw_moves:
            btn = QPushButton(move)
            btn.setFixedSize(40, 40)
            btn.clicked.connect(lambda checked, m=move: self.perform_move(m))
            btn.setToolTip(self.get_move_description(move))
            cw_layout.addWidget(btn)
        moves_layout.addLayout(cw_layout)
        
        # Counterclockwise moves (3)
        ccw_moves = ['F3', 'B3', 'U3', 'D3', 'L3', 'R3']
        ccw_layout = QHBoxLayout()
        for move in ccw_moves:
            btn = QPushButton(move)
            btn.setFixedSize(40, 40)
            btn.clicked.connect(lambda checked, m=move: self.perform_move(m))
            btn.setToolTip(self.get_move_description(move))
            ccw_layout.addWidget(btn)
        moves_layout.addLayout(ccw_layout)
        
        layout.addLayout(moves_layout)

        # Add utility buttons
        utils_layout = QHBoxLayout()
        
        # Reset button
        reset_btn = QPushButton("Reset")
        reset_btn.setFixedSize(80, 40)
        reset_btn.clicked.connect(self.reset_cube)
        reset_btn.setToolTip("Reset cube to initial state")
        utils_layout.addWidget(reset_btn)
        
        # Scramble controls
        scramble_layout = QHBoxLayout()
        scramble_layout.setSpacing(5)
        
        # Number of scramble moves spinbox
        self.scramble_moves = QSpinBox()
        self.scramble_moves.setRange(1, 100)
        self.scramble_moves.setValue(20)
        self.scramble_moves.setFixedSize(50, 40)
        self.scramble_moves.setToolTip("Number of random moves")
        scramble_layout.addWidget(self.scramble_moves)
        
        # Scramble button  
        random_btn = QPushButton("Scramble")
        random_btn.setFixedSize(80, 40)
        random_btn.clicked.connect(self.random_scramble)
        random_btn.setToolTip("Perform random moves")
        scramble_layout.addWidget(random_btn)
        
        utils_layout.addLayout(scramble_layout)
        
        # Add DQN controls
        dqn_layout = QHBoxLayout()
        
        # Training controls layout
        train_controls = QHBoxLayout()
        
        # Min scramble steps spinbox
        min_scramble_label = QLabel("Min Scramble:")
        train_controls.addWidget(min_scramble_label)
        self.min_scramble_steps = QSpinBox()
        self.min_scramble_steps.setRange(1, 100)
        self.min_scramble_steps.setValue(1)
        self.min_scramble_steps.setFixedSize(50, 40)
        self.min_scramble_steps.setToolTip("Minimum number of scramble moves per episode")
        train_controls.addWidget(self.min_scramble_steps)

        # Max scramble steps spinbox
        max_scramble_label = QLabel("Max Scramble:")
        train_controls.addWidget(max_scramble_label)
        self.max_scramble_steps = QSpinBox()
        self.max_scramble_steps.setRange(1, 100)
        self.max_scramble_steps.setValue(3)
        self.max_scramble_steps.setFixedSize(50, 40)
        self.max_scramble_steps.setToolTip("Maximum number of scramble moves per episode")
        train_controls.addWidget(self.max_scramble_steps)
        
        # Max moves spinbox
        max_moves_label = QLabel("Max Moves:")
        train_controls.addWidget(max_moves_label)
        self.max_moves = QSpinBox()
        self.max_moves.setRange(1, 200)
        self.max_moves.setValue(1)
        self.max_moves.setFixedSize(50, 40)
        self.max_moves.setToolTip("Maximum moves allowed before resetting")
        train_controls.addWidget(self.max_moves)

        # Episodes spinbox
        episodes_label = QLabel("Episodes:")
        train_controls.addWidget(episodes_label)
        self.train_episodes = QSpinBox()
        self.train_episodes.setRange(1, 10000)
        self.train_episodes.setValue(1000)
        self.train_episodes.setFixedSize(70, 40)
        self.train_episodes.setToolTip("Number of training episodes")
        train_controls.addWidget(self.train_episodes)
        
        dqn_layout.addLayout(train_controls)
        
        # Train button
        train_btn = QPushButton("Train DQN")
        train_btn.setFixedSize(80, 40)
        train_btn.clicked.connect(self.train_dqn)
        train_btn.setToolTip("Train the DQN solver")
        dqn_layout.addWidget(train_btn)
        
        # Save button
        save_btn = QPushButton("Save Net")
        save_btn.setFixedSize(80, 40)
        save_btn.clicked.connect(lambda: self.solver.save_model('rubiks_dqn.pth'))
        save_btn.setToolTip("Save current network state")
        dqn_layout.addWidget(save_btn)
        
        # Solve button
        solve_btn = QPushButton("Solve")
        solve_btn.setFixedSize(80, 40)
        solve_btn.clicked.connect(self.solve_cube)
        solve_btn.setToolTip("Solve using trained DQN")
        dqn_layout.addWidget(solve_btn)
        
        utils_layout.addLayout(dqn_layout)
        
        layout.addLayout(utils_layout)

        # Add score display
        self.score_label = QLabel()
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setStyleSheet("""
            QLabel { 
                color: white; 
                font-weight: bold;
                font-size: 14px;
                background-color: #444;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.score_label)
        
        # Update all views and score
        self.update_views()
        self.update_score()

    def get_move_description(self, move):
        descriptions = {
            'F1': 'Rotate Front face clockwise',
            'B1': 'Rotate Back face clockwise', 
            'U1': 'Rotate Up face clockwise',
            'D1': 'Rotate Down face clockwise',
            'L1': 'Rotate Left face clockwise',
            'R1': 'Rotate Right face clockwise',
            'F3': 'Rotate Front face counterclockwise',
            'B3': 'Rotate Back face counterclockwise',
            'U3': 'Rotate Up face counterclockwise',
            'D3': 'Rotate Down face counterclockwise',
            'L3': 'Rotate Left face counterclockwise',
            'R3': 'Rotate Right face counterclockwise',
        }
        return descriptions.get(move, 'Perform move ' + move)

    def perform_move(self, move):
        # Validate move against twophase Move enum
        valid_moves = ['F1', 'F3', 'B1', 'B3', 'U1', 'U3', 'D1', 'D3', 'L1', 'L3', 'R1', 'R3',
                      'F2', 'B2', 'U2', 'D2', 'L2', 'R2']
        if move not in valid_moves:
            print(f"Invalid move: {move}")
            return
            
        print(f"\nPerforming move: {move}")
        score_before = self.cube.get_basic_score()
        self.move_history.append(move)  # Add move to history
        
        # Print initial state of faces
        print("Before move:")
        for face in ['U', 'R', 'F', 'D', 'L', 'B']:
            print(f"{face}: {self.cube.faces[face]}")
        
        # Extract face letter and determine direction
        face = move[0]  # First character is face letter
        clockwise = move[-1] == '1'  # Last character is 1 for clockwise, 3 for counterclockwise
        
        self.cube.apply_move(face, clockwise=clockwise)
            
        score_after = self.cube.get_basic_score()
        
        # Print final state of faces
        print("\nAfter move:")
        for face in ['U', 'R', 'F', 'D', 'L', 'B']:
            print(f"{face}: {self.cube.faces[face]}")
            
        print(f"\nScore changed from {score_before}% to {score_after}%")
        
        self.update_views()
        self.update_score()

    def update_views(self):
        for face, view in self.views.items():
            view.update_face(self.cube.faces[face], self.cube.colors)
        self.gl_widget.update()

    def reset_cube(self):
        """Reset the cube to its initial solved state"""
        self.cube.reset()
        self.move_history = []  # Clear move history
        self.update_views()
        self.update_score()

    def get_facelet_string(self):
        """Get the cube state in twophase solver format"""
        # Convert colors to face names according to Color enum
        color_to_face = {}
        for face in self.cube.faces:
            center_color = self.cube.faces[face][1][1]  # Get center color
            color_to_face[center_color] = face
        
        # Build string in correct facelet order per Facelet enum
        faces = ['U', 'R', 'F', 'D', 'L', 'B']  # Color enum order
        facelet_string = ''
        for face in faces:
            face_data = self.cube.faces[face]
            for row in range(3):
                for col in range(3):
                    facelet_string += color_to_face[face_data[row][col]]
        return facelet_string

    def update_score(self):
        """Update the score display"""
        score = self.cube.get_basic_score()
        self.score_label.setText(f"Completion Score: {score}%")
        
        # Convert colors to face names according to Color enum
        # The mapping must match the center colors of each face
        color_to_face = {}
        for face in self.cube.faces:
            center_color = self.cube.faces[face][1][1]  # Get center color
            color_to_face[center_color] = face
        
        # Build string in correct facelet order per Facelet enum
        faces = ['U', 'R', 'F', 'D', 'L', 'B']  # Color enum order
        facelet_string = ''
        for face in faces:
            face_data = self.cube.faces[face]
            # Read face data in Facelet enum order (U1->U9, R1->R9, etc.)
            # Each face is read row by row, left-to-right, top-to-bottom
            for row in range(3):
                for col in range(3):
                    facelet_string += color_to_face[face_data[row][col]]
                    
        print(f"\nCube State: {facelet_string}")
        
        # Print formatted cube state
        print("Current Cube State:")
        for face in ['U', 'R', 'F', 'D', 'L', 'B']:
            print(f"{face}: {self.cube.faces[face]}")
        
        # Send cube state to solver
        try:
            solution = sv.solve(facelet_string, 20, 2)  # max 20 moves, 2 sec timeout
            print(f"Solver solution: {solution}")
        except Exception as e:
            print(f"Solver error: {e}")
    
        
    def is_valid_state(self):
        """Check if current cube state is valid"""
        # Count occurrences of each color
        color_counts = {color: 0 for color in self.cube.colors.keys()}
        
        for face in self.cube.faces.values():
            for row in face:
                for color in row:
                    color_counts[color] += 1
        
        # Each color should appear exactly 9 times
        return all(count == 9 for count in color_counts.values())

    def calculate_reward(self, old_score, new_score):
        # Base reward from score improvement
        reward = (new_score - old_score) / 20.0  # Larger scale for improvements
        
        # Bonus for solving
        if new_score == 100:
            reward += 5.0  # Much larger solving bonus
            
        # Smaller penalty for getting worse
        elif new_score < old_score:
            reward -= 0.1  # Reduced penalty
            
        # Small step penalty to encourage efficiency
        reward -= 0.01
        
        return reward

    def train_dqn(self):
        """Train the DQN solver"""
        episodes = self.train_episodes.value()
        batch_size = 1280  # 10x larger batch size
        success_history = []
        progress = QProgressDialog("Training DQN...", "Cancel", 0, episodes, self)
        progress.setWindowModality(Qt.WindowModal)
        
        # Track recent success rate
        window_size = 100  # Look at last 100 episodes
        
        for episode in range(episodes):
            if progress.wasCanceled():
                break
                
            self.cube.reset()
            
            # Use min/max scramble steps from UI
            scramble_steps = random.randint(
                self.min_scramble_steps.value(),
                self.max_scramble_steps.value()
            )
            
            # Scramble cube with random moves
            for _ in range(scramble_steps):
                move = random.choice(['F1', 'F3', 'B1', 'B3', 'U1', 'U3', 'D1', 'D3', 'L1', 'L3', 'R1', 'R3'])
                face = move[0]
                clockwise = move[1] == '1'
                self.cube.apply_move(face, clockwise)
            
            # Get post-scramble state
            scrambled_state = self.get_facelet_string()
            state = self.solver.get_state(self.cube)
            total_reward = 0
            self.move_history = []
            
            # Use max moves from UI
            for step in range(self.max_moves.value()):
                action = self.solver.get_action(state)
                old_score = self.cube.get_basic_score()
                
                # Apply action
                moves = ['F1', 'F3', 'B1', 'B3', 'U1', 'U3', 
                        'D1', 'D3', 'L1', 'L3', 'R1', 'R3']
                move = moves[action]
                face = move[0]
                clockwise = move[1] == '1'
                self.cube.apply_move(face, clockwise)
                
                # Calculate reward
                new_score = self.cube.get_basic_score()
                reward = self.calculate_reward(old_score, new_score)
                total_reward += reward
                
                next_state = self.solver.get_state(self.cube)
                done = new_score == 100
                
                # Store move and experience
                self.move_history.append(action)
                self.solver.remember(state, action, reward, next_state, done)
                
                # Train on larger batch
                if len(self.solver.memory) >= batch_size:
                    self.solver.replay(batch_size)
                    
                state = next_state
                
                if done:
                    success_history.append(1)
                    current_success_rate = sum(success_history) / len(success_history) * 100
                    print(f"\nEpisode {episode + 1}")
                    print(f"Scramble moves: {scramble_steps} random moves")
                    print(f"Starting state: {scrambled_state}")
                    print(f"Solved in {step + 1} steps with moves: {[moves[a] for a in self.move_history]}")
                    print(f"Score: {old_score:.1f}% -> {new_score:.1f}%")
                    print(f"Total reward: {total_reward:.2f}")
                    print(f"Success rate: {current_success_rate:.1f}%")
                    print("-" * 80)
                    break
            
            if not done:
                success_history.append(0)
            
            # Check success rate and increase max scramble if doing well
            if len(success_history) >= window_size:
                recent_success_rate = sum(success_history[-window_size:]) / window_size * 100
                if recent_success_rate > 90:
                    # Increase max scramble and max moves in UI
                    new_max_scramble = self.max_scramble_steps.value() + 1
                    new_max_moves = self.max_moves.value() + 1
                    self.max_scramble_steps.setValue(new_max_scramble)
                    self.max_moves.setValue(new_max_moves)
                    print(f"\nSuccess rate {recent_success_rate:.1f}% > 90%")
                    print(f"Increased max scramble to {new_max_scramble}")
                    print(f"Increased max moves to {new_max_moves}")
                    # Reset success history when increasing difficulty
                    success_history = []
            
            progress.setValue(episode + 1)
            QApplication.processEvents()
            
        # Calculate final success rate from history
        final_success_rate = sum(success_history) / len(success_history) * 100 if success_history else 0
        
        # Auto-save after training
        self.solver.save_model('rubiks_dqn.pth')
        QMessageBox.information(self, "Training Complete", 
                              f"Training finished!\nSuccess rate: {final_success_rate:.1f}%")
        
    def solve_cube(self):
        """Attempt to solve cube using trained DQN"""
        if not hasattr(self.solver, 'model'):
            QMessageBox.warning(self, "Error", "Please train the DQN first!")
            return
            
        max_moves = 50
        state = self.solver.get_state(self.cube)
        
        for _ in range(max_moves):
            action = self.solver.get_action(state)
            moves = ['F1', 'F3', 'B1', 'B3', 'U1', 'U3', 'D1', 'D3', 'L1', 'L3', 'R1', 'R3']
            move = moves[action]
            
            # Perform move through UI
            self.perform_move(move)
            QApplication.processEvents()
            
            # Check if solved
            score = self.cube.get_basic_score()
            if score == 100:
                QMessageBox.information(self, "Success", "Cube solved!")
                return
                
            state = self.solver.get_state(self.cube)
            
        QMessageBox.warning(self, "Warning", "Could not solve cube in maximum moves!")

    def random_scramble(self):
        """Perform random moves to scramble the cube"""
        import random
        self.cube.reset()  # Reset first
        
        # All possible moves
        moves = ['F1', 'B1', 'U1', 'D1', 'L1', 'R1',
                'F3', 'B3', 'U3', 'D3', 'L3', 'R3']
                
        # Find all move buttons
        move_buttons = {}
        for layout in self.findChildren(QHBoxLayout):
            for i in range(layout.count()):
                widget = layout.itemAt(i).widget()
                if isinstance(widget, QPushButton) and widget.text() in moves:
                    move_buttons[widget.text()] = widget
        
        # Perform random moves by clicking buttons
        for _ in range(self.scramble_moves.value()):
            move = random.choice(moves)
            if move in move_buttons:
                move_buttons[move].click()
                # Small delay to allow UI updates
                QApplication.processEvents()

def headless_train(min_scramble, max_scramble, max_moves, episodes):
    """Run headless DQN training with specified parameters"""
    cube = RubiksCube()
    solver = RubiksSolver()
    try:
        solver.load_model('rubiks_dqn.pth')
        print("Loaded existing DQN model:")
        print(solver.model)
    except:
        print("Starting with new DQN model:")
        print(solver.model)
        
    success_history = []
    window_size = 100
    batch_size = 1280
    
    for episode in range(episodes):
        cube.reset()
        scramble_steps = random.randint(min_scramble, max_scramble)
        
        # Scramble cube
        for _ in range(scramble_steps):
            move = random.choice(['F1', 'F3', 'B1', 'B3', 'U1', 'U3', 'D1', 'D3', 'L1', 'L3', 'R1', 'R3'])
            face = move[0]
            clockwise = move[1] == '1'
            cube.apply_move(face, clockwise)
        
        scrambled_state = ''
        for f in ['U', 'R', 'F', 'D', 'L', 'B']:
            for row in cube.faces[f]:
                for col in row:
                    if col == 'white': scrambled_state += 'U'
                    elif col == 'red': scrambled_state += 'R'
                    elif col == 'green': scrambled_state += 'F'
                    elif col == 'yellow': scrambled_state += 'D'
                    elif col == 'orange': scrambled_state += 'L'
                    elif col == 'blue': scrambled_state += 'B'
                    
        state = solver.get_state(cube)
        total_reward = 0
        move_history = []
        
        for step in range(max_moves):
            action = solver.get_action(state)
            old_score = cube.get_basic_score()
            
            moves = ['F1', 'F3', 'B1', 'B3', 'U1', 'U3', 'D1', 'D3', 'L1', 'L3', 'R1', 'R3']
            move = moves[action]
            face = move[0]
            clockwise = move[1] == '1'
            cube.apply_move(face, clockwise)
            
            new_score = cube.get_basic_score()
            reward = (new_score - old_score) / 20.0
            if new_score == 100:
                reward += 5.0
            elif new_score < old_score:
                reward -= 0.1
            reward -= 0.01
            
            total_reward += reward
            
            next_state = solver.get_state(cube)
            done = new_score == 100
            
            move_history.append(action)
            solver.remember(state, action, reward, next_state, done)
            
            if len(solver.memory) >= batch_size:
                solver.replay(batch_size)
                
            state = next_state
            
            if done:
                success_history.append(1)
                current_success_rate = sum(success_history[-window_size:]) / len(success_history[-window_size:]) * 100
                print(f"\nEpisode {episode + 1}")
                print(f"Scramble moves: {scramble_steps} random moves")
                print(f"Starting state: {scrambled_state}")
                print(f"Solved in {step + 1} steps with moves: {[moves[a] for a in move_history]}")
                print(f"Score: {old_score:.1f}% -> {new_score:.1f}%")
                print(f"Total reward: {total_reward:.2f}")
                print(f"Success rate: {current_success_rate:.1f}%")
                print("-" * 80)
                break
        
        if not done:
            success_history.append(0)
            
        if len(success_history) >= window_size:
            recent_success_rate = sum(success_history[-window_size:]) / window_size * 100
            if recent_success_rate > 90:
                max_scramble += 1
                max_moves += 1
                print(f"\nSuccess rate {recent_success_rate:.1f}% > 90%")
                print(f"Increased max scramble to {max_scramble}")
                print(f"Increased max moves to {max_moves}")
                success_history = []
                
    solver.save_model('rubiks_dqn.pth')
    final_success_rate = sum(success_history) / len(success_history) * 100 if success_history else 0
    print(f"\nTraining complete! Final success rate: {final_success_rate:.1f}%")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Rubik\'s Cube DQN Trainer')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--min-scramble', type=int, default=1, help='Minimum scramble moves')
    parser.add_argument('--max-scramble', type=int, default=3, help='Maximum scramble moves')
    parser.add_argument('--max-moves', type=int, default=50, help='Maximum moves per solve attempt')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    
    args = parser.parse_args()
    
    if args.headless:
        headless_train(args.min_scramble, args.max_scramble, args.max_moves, args.episodes)
    else:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
