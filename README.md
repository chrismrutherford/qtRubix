# qtRubix

A 3D Rubik's Cube simulator built with PyQt5 and OpenGL.

<img src="https://github.com/chrismrutherford/qtRubix/blob/main/qtRubix.png" alt="Rubik's Cube Simulator" width="600"/>

## Features

- Interactive 3D cube visualization with mouse rotation
- 2D face views for all sides
- Support for standard cube moves (F, B, U, D, L, R)
- Middle layer moves (M, E, S)
- Real-time updates across all views

## Requirements

- Python 3.x
- PyQt5
- PyOpenGL

## Installation

```bash
pip install PyQt5 PyOpenGL
```

## Usage

Run the simulator:

```bash
python qtRubix.py
```

### Controls

- Click and drag on the 3D view to rotate the cube
- Use the buttons to perform moves:
  - F: Front face clockwise
  - B: Back face clockwise
  - U: Up face clockwise
  - D: Down face clockwise
  - L: Left face clockwise
  - R: Right face clockwise
  - M: Middle layer (between L and R)
  - E: Equatorial layer (between U and D)
  - S: Standing layer (between F and B)
