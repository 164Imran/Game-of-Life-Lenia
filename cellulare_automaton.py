import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
import numpy as np

# ============================================================================
# LENIA: Continuous Cellular Automaton
# ============================================================================
# Implementation of Lenia, a generalization of Conway's Game of Life
# Reference: Bert Wang-Chak Chan (2019)
# ============================================================================

# --- Unused/Incomplete Function (Consider removing) ---
def up_date(N):
    """
    INCOMPLETE: This function appears to be a stub and doesn't perform any operation.
    Consider removing if not needed.
    """
    n = len(N)
    for i in range(n):
        for j in range(n):
            if i >= 2 and j >= 2:
                pass  # No operation performed


# --- Kernel Generation ---
def create_circular_kernel(R, smooth=True):
    """
    Creates a circular convolution kernel for Lenia neighborhood detection.
    
    Parameters:
    -----------
    R : int
        Radius of the circular kernel
    smooth : bool, optional
        If True, creates a smooth kernel (currently creates a binary circular mask)
        
    Returns:
    --------
    numpy.ndarray
        Normalized 2D kernel of shape (2*R+1, 2*R+1)
        
    Notes:
    ------
    - The kernel defines which cells are considered "neighbors" in the Lenia automaton
    - Currently implements a binary circular mask (all cells within radius R have equal weight)
    - Could be extended to implement distance-based weighting for smoother behavior
    """
    d = 2*R + 1  # Kernel diameter
    kernel = np.zeros((d, d))
    center = R
    
    # Create circular mask
    for i in range(d):
        for j in range(d):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if smooth:
                # Binary circular kernel: 1 if within radius, 0 otherwise
                if dist <= R:
                    kernel[i, j] = 1
    
    # Normalize so kernel sums to 1 (required for proper convolution)
    kernel = kernel / np.sum(kernel)
    return kernel


# --- Lenia Parameters ---
kernel = create_circular_kernel(7)  # Neighborhood radius of 7 cells

# Growth function parameters (control how cells grow/decay)
mu = 0.09        # Optimal neighborhood density for growth
sigma = 0.015    # Width of the growth response curve
dt = 0.01        # Time step for continuous update


# --- Core Lenia Functions ---
def conv(N):
    """
    Computes the neighborhood density at each cell using convolution.
    
    Parameters:
    -----------
    N : numpy.ndarray
        Current state grid (values between 0 and 1)
        
    Returns:
    --------
    numpy.ndarray
        Convolved grid representing local neighborhood densities
        
    Notes:
    ------
    - Uses 'wrap' boundary for toroidal topology (edges wrap around)
    - Returns average density of neighbors within kernel radius
    """
    return signal.convolve2d(N, kernel, mode='same', boundary='wrap')


def G(N):
    """
    Growth function: determines how cells grow or decay based on neighborhood density.
    
    Parameters:
    -----------
    N : numpy.ndarray
        Neighborhood density values (output from convolution)
        
    Returns:
    --------
    numpy.ndarray
        Growth rate at each cell (positive = growth, negative = decay)
        
    Notes:
    ------
    - Implements a Gaussian-like growth curve centered at mu
    - Cells grow optimally when neighborhood density ≈ mu
    - Returns values in range [-1, 1]
    """
    return 2*np.exp(-(N-mu)**2 / (2*sigma**2)) - 1


def update_lenia(N, dt):
    """
    Updates the Lenia state for one time step using Euler integration.
    
    Parameters:
    -----------
    N : numpy.ndarray
        Current state grid
    dt : float
        Time step size
        
    Returns:
    --------
    numpy.ndarray
        Updated state grid, clipped to [0, 1]
        
    Notes:
    ------
    - Implements: N(t+dt) = N(t) + dt * G(conv(N(t)))
    - Clips values to ensure they stay in valid range [0, 1]
    """
    return np.clip(N + dt*G(conv(N)), 0, 1)


# ============================================================================
# INITIALIZATION: Create "Orbium" Pattern
# ============================================================================
# Orbium is a stable glider-like pattern in Lenia that moves smoothly

# Initialize grid
N = 256  # Grid width (currently unused - state is 100x100)
M = 256  # Grid height (currently unused - state is 100x100)

# Create Gaussian blob centered at (50, 50)
center_x, center_y = 50, 50
y, x = np.ogrid[:100, :100]
dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)

# CRITICAL: Amplitude of 0.5 creates stable Orbium pattern
# - Different amplitudes produce different behaviors (stable/growing/dying)
# - Gaussian width of 10 cells matches the kernel radius
state = 0.5 * np.exp(-(dist_from_center**2) / (2 * 10**2))


# ============================================================================
# VISUALIZATION & ANIMATION
# ============================================================================

# Setup matplotlib figure
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(state, cmap='inferno', interpolation='bilinear', vmin=0, vmax=1)
ax.set_title("Lenia - Orbium")
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)


def animate(frame):
    """
    Animation update function called for each frame.
    
    Parameters:
    -----------
    frame : int
        Current frame number (provided by FuncAnimation)
        
    Returns:
    --------
    list
        List of artists to be redrawn (required for blit=True)
        
    Notes:
    ------
    - Updates state using Lenia rules
    - Could run multiple steps per frame for faster evolution
    """
    global state
    state = update_lenia(state, dt)
    im.set_array(state)
    ax.set_title(f"Lenia - Frame {frame}")  # Added frame counter
    return [im]


# Create animation
# - 1000 frames total
# - 50ms between frames (20 FPS)
# - blit=True for better performance (only redraw changed parts)
ani = FuncAnimation(fig, animate, frames=1000, interval=50, blit=True)

plt.show()