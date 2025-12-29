import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Dimensions de la grille
LARGEUR, HAUTEUR = 50, 50

# Initialiser la grille aléatoirement
grille = np.random.choice([0, 1], size=(HAUTEUR, LARGEUR), p=[0.8, 0.2])

def compter_voisins(grille):
    """Compte le nombre de voisins vivants pour chaque cellule"""
    voisins = np.zeros_like(grille)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            voisins += np.roll(np.roll(grille, i, axis=0), j, axis=1)
    return voisins

def evolution(grille):
    """Applique les règles du jeu de la vie"""
    voisins = compter_voisins(grille)
    
    # Une cellule vivante avec 2 ou 3 voisins survit
    survie = (grille == 1) & ((voisins == 2) | (voisins == 3))
    
    # Une cellule morte avec exactement 3 voisins devient vivante
    naissance = (grille == 0) & (voisins == 3)
    
    nouvelle_grille = survie | naissance
    return nouvelle_grille.astype(int)

# Visualisation
fig, ax = plt.subplots()
img = ax.imshow(grille, cmap='binary', interpolation='nearest')

def update(frame):
    global grille
    grille = evolution(grille)
    img.set_data(grille)
    return [img]

anim = FuncAnimation(fig, update, frames=200, interval=100, blit=True)
plt.show()
