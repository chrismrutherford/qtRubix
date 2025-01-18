from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QGraphicsView, QGraphicsScene, QApplication,
                            QLabel, QOpenGLWidget, QProgressDialog, QCheckBox,
                            QMessageBox, QSpinBox)
from dqn_solver import RubiksCubeSolver
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtCore import Qt, QTimer
from OpenGL.GL import *
from OpenGL.GLU import *
import sys
import math

class RubiksCube:
    def __init__(self):
        self.scramble_steps = 4  # Default scramble steps
        self.solve_steps = 20    # Default solve steps
        self.default_faces = {
            'F': [['red']*3 for _ in range(3)],
            'B': [['orange']*3 for _ in range(3)],
            'U': [['white']*3 for _ in range(3)],
            'D': [['yellow']*3 for _ in range(3)],
            'L': [['green']*3 for _ in range(3)],
            'R': [['blue']*3 for _ in range(3)]
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
        """Calculate basic completion score based on face matches"""
        score = 0
        max_score = 54  # 9 squares per face * 6 faces
        
        # Check each square against its face's center color
        for face in self.faces:
            center = self.faces[face][1][1]  # Center color of this face
            for row in self.faces[face]:
                for square in row:
                    if square == center:
                        score += 1

        # Convert to percentage
        return round((score / max_score) * 100, 1)
        
    def get_advanced_score(self):
        """Calculate entropy-based score for cube state"""
        import math
        
        # Calculate color distribution entropy for each face
        total_entropy = 0
        for face in self.faces:
            # Count occurrences of each color on this face
            color_counts = {}
            total_squares = 9  # 3x3 face
            
            for row in self.faces[face]:
                for color in row:
                    color_counts[color] = color_counts.get(color, 0) + 1
            
            # Calculate entropy for this face
            face_entropy = 0
            for count in color_counts.values():
                p = count / total_squares
                face_entropy -= p * math.log2(p)
            
            total_entropy += face_entropy
        
        # Maximum entropy would be when colors are evenly distributed
        # (log2(6) ≈ 2.58 per face, times 6 faces)
        max_entropy = 6 * math.log2(6)
        
        # Convert to percentage where 100% means solved (minimum entropy)
        # and 0% means maximum disorder (maximum entropy)
        score = (1 - (total_entropy / max_entropy)) * 100
        
        return round(score, 1)

    def rotate_face_clockwise(self, face):
        # Rotate the selected face clockwise
        self.faces[face] = [list(row) for row in zip(*self.faces[face][::-1])]
        
        # Define the edges that need to be rotated for each face
        if face == 'F':
            # Save the top edge
            temp = self.faces['U'][0].copy()
            # Move left edge to top
            self.faces['U'][0][0] = self.faces['L'][2][2]
            self.faces['U'][0][1] = self.faces['L'][1][2]
            self.faces['U'][0][2] = self.faces['L'][0][2]
            # Move bottom edge to left
            self.faces['L'][0][2] = self.faces['D'][2][2]
            self.faces['L'][1][2] = self.faces['D'][2][1]
            self.faces['L'][2][2] = self.faces['D'][2][0]
            # Move right edge to bottom
            self.faces['D'][2][0] = self.faces['R'][2][0]
            self.faces['D'][2][1] = self.faces['R'][1][0]
            self.faces['D'][2][2] = self.faces['R'][0][0]
            # Move saved top edge to right
            self.faces['R'][0][0] = temp[2]
            self.faces['R'][1][0] = temp[1]
            self.faces['R'][2][0] = temp[0]
            
        elif face == 'B':
            # Save the top edge
            temp = self.faces['U'][2].copy()
            # Move right edge to top
            self.faces['U'][2][0] = self.faces['R'][0][2]
            self.faces['U'][2][1] = self.faces['R'][1][2]
            self.faces['U'][2][2] = self.faces['R'][2][2]
            # Move down edge to right
            self.faces['R'][0][2] = self.faces['D'][0][2]
            self.faces['R'][1][2] = self.faces['D'][0][1]
            self.faces['R'][2][2] = self.faces['D'][0][0]
            # Move left edge to down
            self.faces['D'][0][0] = self.faces['L'][2][0]
            self.faces['D'][0][1] = self.faces['L'][1][0]
            self.faces['D'][0][2] = self.faces['L'][0][0]
            # Move saved top edge to left
            self.faces['L'][0][0] = temp[2]
            self.faces['L'][1][0] = temp[1]
            self.faces['L'][2][0] = temp[0]
            
        elif face == 'L':
            # Save the front edge
            temp = [self.faces['F'][i][0] for i in range(3)]
            # Move top edge to front
            for i in range(3):
                self.faces['F'][i][0] = self.faces['U'][i][0]
            # Move back edge to top (reversed)
            for i in range(3):
                self.faces['U'][i][0] = self.faces['B'][2-i][2]
            # Move bottom edge to back (reversed)
            for i in range(3):
                self.faces['B'][i][2] = self.faces['D'][2-i][0]
            # Move saved front edge to bottom
            for i in range(3):
                self.faces['D'][i][0] = temp[i]
                
        elif face == 'R':
            # Save the front edge
            temp = [self.faces['F'][i][2] for i in range(3)]
            # Move bottom edge to front
            for i in range(3):
                self.faces['F'][i][2] = self.faces['D'][i][2]
            # Move back edge to bottom (reversed)
            for i in range(3):
                self.faces['D'][i][2] = self.faces['B'][2-i][0]
            # Move top edge to back (reversed)
            for i in range(3):
                self.faces['B'][i][0] = self.faces['U'][2-i][2]
            # Move saved front edge to top
            for i in range(3):
                self.faces['U'][i][2] = temp[i]
                
        elif face == 'U':
            # Save the front edge
            temp = self.faces['F'][0].copy()
            # Move right edge to front
            self.faces['F'][0] = self.faces['R'][0]
            # Move back edge to right
            self.faces['R'][0] = self.faces['B'][0]
            # Move left edge to back
            self.faces['B'][0] = self.faces['L'][0]
            # Move saved front edge to left
            self.faces['L'][0] = temp
            
        elif face == 'D':
            # Save the front edge
            temp = self.faces['F'][2].copy()
            # Move left edge to front
            self.faces['F'][2] = self.faces['L'][2]
            # Move back edge to left
            self.faces['L'][2] = self.faces['B'][2]
            # Move right edge to back
            self.faces['B'][2] = self.faces['R'][2]
            # Move saved front edge to right
            self.faces['R'][2] = temp


    def rotate_M(self):
        # Middle layer between L and R faces
        for i in range(3):
            temp = self.faces['F'][i][1]
            self.faces['F'][i][1] = self.faces['U'][i][1]
            self.faces['U'][i][1] = self.faces['B'][i][1]
            self.faces['B'][i][1] = self.faces['D'][i][1]
            self.faces['D'][i][1] = temp

    def rotate_E(self):
        # Equatorial layer between U and D faces
        temp = self.faces['F'][1]
        self.faces['F'][1] = self.faces['R'][1]
        self.faces['R'][1] = self.faces['B'][1]
        self.faces['B'][1] = self.faces['L'][1]
        self.faces['L'][1] = temp

    def rotate_S(self):
        # Standing layer between F and B faces
        for i in range(3):
            temp = self.faces['U'][1][i]
            self.faces['U'][1][i] = self.faces['L'][2-i][1]
            self.faces['L'][2-i][1] = self.faces['D'][1][2-i]
            self.faces['D'][1][2-i] = self.faces['R'][i][1]
            self.faces['R'][i][1] = temp

    def rotate_face_counterclockwise(self, face):
        # Rotate face counterclockwise by doing 3 clockwise rotations
        for _ in range(3):
            self.rotate_face_clockwise(face)

    def rotate_M_ccw(self):
        # Counterclockwise middle layer rotation
        for _ in range(3):
            self.rotate_M()

    def rotate_E_ccw(self):
        # Counterclockwise equatorial layer rotation
        for _ in range(3):
            self.rotate_E()

    def rotate_S_ccw(self):
        # Counterclockwise standing layer rotation
        for _ in range(3):
            self.rotate_S()

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
        # Initialize solver at startup
        self.solver = RubiksCubeSolver(self.cube)
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.blink_timeout)
        self.blink_count = 0
        self.original_style = None
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

        # Create 2D cube views
        faces_layout = QHBoxLayout()
        self.views = {}
        for face in ['F', 'B', 'U', 'D', 'L', 'R']:
            view = CubeFaceView(face)
            self.views[face] = view
            faces_layout.addWidget(view)
        layout.addLayout(faces_layout)

        # Create move buttons
        moves_layout = QVBoxLayout()
        
        # Clockwise moves
        cw_moves = ['F', 'B', 'U', 'D', 'L', 'R', 'M', 'E', 'S']
        cw_layout = QHBoxLayout()
        for move in cw_moves:
            btn = QPushButton(move)
            btn.setFixedSize(40, 40)
            btn.clicked.connect(lambda checked, m=move: self.perform_move(m))
            btn.setToolTip(self.get_move_description(move))
            cw_layout.addWidget(btn)
        moves_layout.addLayout(cw_layout)
        
        # Counterclockwise moves
        ccw_moves = ["F'", "B'", "U'", "D'", "L'", "R'", "M'", "E'", "S'"]
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
        
        # Scramble controls
        scramble_layout = QHBoxLayout()
        random_btn = QPushButton("Random")
        random_btn.setFixedSize(80, 40)
        random_btn.clicked.connect(self.random_scramble)
        random_btn.setToolTip("Perform random moves")
        scramble_layout.addWidget(random_btn)
        
        self.scramble_spin = QSpinBox()
        self.scramble_spin.setRange(1, 1000000)
        self.scramble_spin.setValue(4)
        self.scramble_spin.setToolTip("Number of scramble moves for training")
        self.scramble_spin.valueChanged.connect(self.update_scramble_steps)
        scramble_layout.addWidget(self.scramble_spin)
        
        utils_layout.addLayout(scramble_layout)
        
        # Reset button
        reset_btn = QPushButton("Reset")
        reset_btn.setFixedSize(80, 40)
        reset_btn.clicked.connect(self.reset_cube)
        reset_btn.setToolTip("Reset cube to initial state")
        utils_layout.addWidget(reset_btn)
        
        # Solve controls
        solve_layout = QHBoxLayout()
        solve_btn = QPushButton("Solve")
        solve_btn.setFixedSize(80, 40)
        solve_btn.setEnabled(True)
        solve_btn.clicked.connect(self.solve_cube)
        solve_btn.setToolTip("Attempt to solve cube")
        solve_layout.addWidget(solve_btn)
        
        self.solve_spin = QSpinBox()
        self.solve_spin.setRange(1, 1000000)
        self.solve_spin.setValue(20)
        self.solve_spin.setToolTip("Maximum solve moves for training")
        self.solve_spin.valueChanged.connect(self.update_solve_steps)
        solve_layout.addWidget(self.solve_spin)
        
        utils_layout.addLayout(solve_layout)
        
        # Train button (placeholder)
        train_btn = QPushButton("Train")
        train_btn.setFixedSize(80, 40)
        train_btn.setToolTip("Train DQN solver")
        train_btn.clicked.connect(self.train_solver)
        utils_layout.addWidget(train_btn)
        
        # Add save button
        save_btn = QPushButton("Save Network")
        save_btn.clicked.connect(self.save_network)
        save_btn.setToolTip("Save the current network weights")
        utils_layout.addWidget(save_btn)
        
        # Add real-time update checkbox
        self.realtime_updates = QCheckBox("Real-time Updates")
        self.realtime_updates.setChecked(True)
        self.realtime_updates.setToolTip("Enable/disable real-time UI updates during training")
        utils_layout.addWidget(self.realtime_updates)
        
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
            'F': 'Rotate Front face clockwise',
            'B': 'Rotate Back face clockwise',
            'U': 'Rotate Up face clockwise',
            'D': 'Rotate Down face clockwise',
            'L': 'Rotate Left face clockwise',
            'R': 'Rotate Right face clockwise',
            'M': 'Rotate Middle layer (between L and R)',
            'E': 'Rotate Equatorial layer (between U and D)',
            'S': 'Rotate Standing layer (between F and B)',
            "F'": 'Rotate Front face counterclockwise',
            "B'": 'Rotate Back face counterclockwise',
            "U'": 'Rotate Up face counterclockwise',
            "D'": 'Rotate Down face counterclockwise',
            "L'": 'Rotate Left face counterclockwise',
            "R'": 'Rotate Right face counterclockwise',
            "M'": 'Rotate Middle layer counterclockwise',
            "E'": 'Rotate Equatorial layer counterclockwise',
            "S'": 'Rotate Standing layer counterclockwise',
        }
        return descriptions.get(move, 'Perform move ' + move)

    def perform_move(self, move):
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
        # Counterclockwise moves
        elif move == "U'":
            self.cube.rotate_face_counterclockwise('D')
        elif move == "D'":
            self.cube.rotate_face_counterclockwise('U')
        elif move[-1] == "'" and move[0] in ['F', 'B', 'L', 'R']:
            self.cube.rotate_face_counterclockwise(move[0])
        elif move == "M'":
            self.cube.rotate_M_ccw()
        elif move == "E'":
            self.cube.rotate_E_ccw()
        elif move == "S'":
            self.cube.rotate_S_ccw()
        self.update_views()
        self.update_score()

    def update_views(self):
        for face, view in self.views.items():
            view.update_face(self.cube.faces[face], self.cube.colors)
        self.gl_widget.update()

    def random_scramble(self):
        """Perform random moves to scramble the cube"""
        import random
        self.cube.reset()  # Reset first
        self.moves_remaining = self.scramble_spin.value()
        self.timer = QTimer()
        self.timer.timeout.connect(self.perform_random_move)
        self.timer.start(10)  # 10ms between moves
        
    def perform_random_move(self):
        """Perform a single random move"""
        import random
        moves = ['F', 'B', 'U', 'D', 'L', 'R', 'M', 'E', 'S',
                "F'", "B'", "U'", "D'", "L'", "R'", "M'", "E'", "S'"]
        move = random.choice(moves)
        self.perform_move(move)
        
        self.moves_remaining -= 1
        if self.moves_remaining <= 0:
            self.timer.stop()

    def reset_cube(self):
        """Reset the cube to its initial solved state"""
        self.cube.reset()
        self.update_views()
        self.update_score()

    def update_score(self):
        """Update the score display"""
        basic_score = self.cube.get_basic_score()
        entropy_score = self.cube.get_advanced_score()
        self.score_label.setText(f"Basic Score: {basic_score}% | Entropy Score: {entropy_score}%")
        
    def blink_success(self):
        """Start blinking animation for success"""
        if self.original_style is None:
            self.original_style = self.score_label.styleSheet()
        self.blink_count = 0
        self.blink_timer.start(250)  # Blink every 250ms
        
    def blink_timeout(self):
        """Handle each blink interval"""
        self.blink_count += 1
        if self.blink_count > 6:  # 3 full blinks
            self.blink_timer.stop()
            self.score_label.setStyleSheet(self.original_style)
            return
            
        if self.blink_count % 2 == 0:
            self.score_label.setStyleSheet(self.original_style)
        else:
            self.score_label.setStyleSheet("""
                QLabel { 
                    color: black; 
                    font-weight: bold;
                    font-size: 14px;
                    background-color: #00ff00;
                    padding: 5px;
                    border-radius: 3px;
                }
            """)
        
    def save_network(self):
        """Save the current network weights"""
        if hasattr(self, 'solver'):
            self.solver.save_model()
            QMessageBox.information(self, "Success", "Network saved successfully!")
        else:
            QMessageBox.warning(self, "Error", "No trained network to save!")
            
    def solve_cube(self):
        """Attempt to solve the cube using the trained solver"""
        solution, reward, scrambled_state, state_history = self.solver.solve(max_steps=self.solve_spin.value())
        if solution:
            self.blink_success()  # Trigger success animation
            
            # Build detailed solution message showing each move and resulting state
            message = f"Solution found with {len(solution)} moves!\n\n"
            message += f"Initial Scrambled State:\n{scrambled_state}\n\n"
            message += "Solution Steps:\n"
            
            # Add each move and the resulting state
            for i, (move, state) in enumerate(zip(solution, state_history[1:]), 1):
                message += f"\nStep {i}: {move}\n"
                message += f"State after move:\n{state}\n"
                message += "-" * 40  # Separator line
            QMessageBox.information(self, "Solution Details", message)
        else:
            QMessageBox.warning(self, "No Solution", "Could not find solution within move limit")
            
    def update_scramble_steps(self, value):
        """Update cube's scramble steps setting"""
        self.cube.scramble_steps = value
        
    def update_solve_steps(self, value):
        """Update cube's solve steps setting"""
        self.cube.solve_steps = value
        
    def train_solver(self):
        """Train the DQN solver"""
        print("\nStarting training...")
        
        def update_ui():
            """Update UI and process events if real-time updates are enabled"""
            self.update_views()
            self.update_score()
            QApplication.processEvents()
        
        # Create progress dialog
        progress = QProgressDialog("Training DQN solver...", "Cancel", 0, 1000000, self)
        progress.setWindowTitle("Training Progress")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        
        def training_callback():
            """Combined callback that handles successful solves"""
            if self.realtime_updates.isChecked():
                update_ui()  # Update UI if real-time enabled
                self.blink_success()  # Trigger blink animation on success
        
        # Train in batches to update UI
        training_iterations = 1000000
        for i in range(training_iterations):
            if progress.wasCanceled():
                break
            self.solver.train(callback=training_callback)
            progress.setValue(i)
            QApplication.processEvents()
        
        progress.setValue(training_iterations)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
