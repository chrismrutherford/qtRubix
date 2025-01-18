from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QGraphicsView, QGraphicsScene, QApplication,
                            QLabel, QOpenGLWidget)
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLU import *
import sys
import math

class RubiksCube:
    def __init__(self):
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
        self.faces = {
            'F': [['red']*3 for _ in range(3)],
            'B': [['orange']*3 for _ in range(3)],
            'U': [['white']*3 for _ in range(3)],
            'D': [['yellow']*3 for _ in range(3)],
            'L': [['green']*3 for _ in range(3)],
            'R': [['blue']*3 for _ in range(3)]
        }

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
        
        # Update all views
        self.update_views()

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

    def update_views(self):
        for face, view in self.views.items():
            view.update_face(self.cube.faces[face], self.cube.colors)
        self.gl_widget.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
