# Lenia - Continuous Cellular Automaton

A Python implementation of **Lenia**, a continuous generalization of Conway's Game of Life that produces organic, life-like patterns and behaviors.

![Lenia Animation](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.7+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 About

Lenia is a family of continuous cellular automata discovered by Bert Wang-Chak Chan in 2019. Unlike traditional cellular automata with discrete states (alive/dead), Lenia uses continuous values between 0 and 1, resulting in smooth, organic movements that resemble biological life forms.

This implementation showcases the **Orbium** pattern, a stable glider-like structure that moves gracefully across the grid.

### Key Features

- **Continuous States**: Cells have values between 0 and 1 instead of binary states
- **Smooth Evolution**: Uses Euler integration for continuous time steps
- **Circular Neighborhoods**: Implements distance-based neighbor detection
- **Toroidal Topology**: Edges wrap around for infinite space simulation
- **Real-time Visualization**: Animated display using matplotlib

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.7
numpy
scipy
matplotlib
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lenia.git
cd lenia
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Simulation

```bash
python lenia.py
```

The animation window will open showing the Orbium pattern evolving in real-time.

## 🎮 Usage

### Basic Example

```python
from lenia import create_circular_kernel, update_lenia
import numpy as np

# Create initial state (Gaussian blob)
state = 0.5 * np.exp(-(distance_grid**2) / (2 * 10**2))

# Update simulation
for step in range(1000):
    state = update_lenia(state, dt=0.01)
```

### Customizing Parameters

Modify these parameters to explore different behaviors:

```python
# Growth function parameters
mu = 0.09        # Optimal neighborhood density (0.0 - 1.0)
sigma = 0.015    # Growth curve width (smaller = sharper)
dt = 0.01        # Time step (smaller = smoother but slower)

# Kernel parameters
kernel_radius = 7  # Neighborhood size (larger = more interaction)
```

### Creating Different Patterns

Experiment with initial conditions:

```python
# Random initialization
state = np.random.random((100, 100)) * 0.5

# Multiple blobs
state1 = 0.5 * np.exp(-((x-30)**2 + (y-30)**2) / 200)
state2 = 0.5 * np.exp(-((x-70)**2 + (y-70)**2) / 200)
state = state1 + state2
```

## 🧬 How Lenia Works

### The Core Algorithm

1. **Convolution**: Calculate neighborhood density for each cell
   ```
   U(t) = K ⊛ N(t)
   ```

2. **Growth Function**: Determine growth/decay rate based on density
   ```
   G(u) = 2 * exp(-(u - μ)² / (2σ²)) - 1
   ```

3. **Update State**: Apply growth using Euler integration
   ```
   N(t + dt) = clip(N(t) + dt * G(U(t)), 0, 1)
   ```

### Parameter Guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| `mu` | 0.0 - 1.0 | Optimal density for growth (higher = denser patterns) |
| `sigma` | 0.001 - 0.1 | Tolerance range (smaller = more selective) |
| `dt` | 0.001 - 0.1 | Evolution speed (smaller = smoother but slower) |
| `kernel_radius` | 3 - 20 | Interaction range (larger = long-range effects) |

### Known Stable Patterns

- **Orbium**: Smooth glider (current implementation)
- **Hydrogeminium**: Rotating blob
- **Scutium**: Shield-like structure
- **Gyrorbium**: Spiral patterns

## 🎨 Visualization Options

### Change Colormap

```python
im = ax.imshow(state, cmap='viridis', ...)  # Try: 'plasma', 'magma', 'cividis'
```

### Adjust Animation Speed

```python
ani = FuncAnimation(fig, animate, frames=1000, interval=30, ...)  # Lower = faster
```

### Save Animation

```python
from matplotlib.animation import PillowWriter
writer = PillowWriter(fps=20)
ani.save('lenia.gif', writer=writer)
```

## 🔬 Mathematical Background

Lenia extends Conway's Game of Life by:

1. **Continuous Space**: Cells have real values ∈ [0, 1] instead of {0, 1}
2. **Continuous Time**: Uses differential equations instead of discrete steps
3. **Continuous Neighborhoods**: Smooth distance-based kernels instead of fixed grids

The general form:
```
∂N/∂t = G(K ⊛ N)
```

Where:
- `N`: State field (cell values)
- `K`: Convolution kernel (neighborhood definition)
- `G`: Growth function (update rule)
- `⊛`: Convolution operator

## 📚 References

- **Original Paper**: Chan, B. W.-C. (2019). "Lenia - Biology of Artificial Life". *Complex Systems*, 28(3).
- **Website**: [https://chakazul.github.io/lenia.html](https://chakazul.github.io/lenia.html)
- **Video**: [Lenia - Mathematical Life Forms](https://www.youtube.com/watch?v=iE46jKYcI4Y)

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- Implement additional growth functions (polynomial, step-wise)
- Add preset patterns library
- Create interactive parameter tuning
- Optimize performance with GPU acceleration
- Add 3D Lenia visualization

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/lenia.git

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and commit
git commit -m "Add amazing feature"

# Push and create pull request
git push origin feature/amazing-feature
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Bert Wang-Chak Chan** - Creator of Lenia
- **Conway's Game of Life** - Original inspiration
- **SciPy** - Convolution implementation
- **Matplotlib** - Visualization tools

## 📧 Contact

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Email**: your.email@example.com
- **Twitter**: [@yourhandle](https://twitter.com/yourhandle)

## 🗺️ Roadmap

- [ ] Add command-line interface for parameter control
- [ ] Implement pattern library with presets
- [ ] Create Jupyter notebook tutorial
- [ ] Add GPU acceleration (CUDA/OpenCL)
- [ ] Build web-based interactive demo
- [ ] Implement 3D Lenia
- [ ] Add pattern evolution tracking
- [ ] Create pattern taxonomy classifier

---

**Star ⭐ this repository if you find it interesting!**

Made with ❤️ and mathematics
