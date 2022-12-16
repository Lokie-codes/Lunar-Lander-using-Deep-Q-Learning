import pygame
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Constants
GRAVITY = 0.2
THRUST = 0.5
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800

# Initialize Pygame
pygame.init()

# Set up the display window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Lunar Lander")

# Load the lunar lander sprite
lander = pygame.image.load("lander.png")

# Set up the game clock
clock = pygame.time.Clock()

# Initialize the Q learning model
model = Sequential()
model.add(Dense(24, input_shape=(4,), activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(2, activation="linear"))
model.compile(loss="mse", optimizer=Adam(lr=0.001))

# Initialize the game state
done = False
x = SCREEN_WIDTH / 2
y = SCREEN_HEIGHT / 2
vx = 0
vy = 0
fuel = 100
score = 0

# Main game loop
while not done:
    # Handle user input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                if fuel > 0:
                    vy -= THRUST
                    fuel -= 1
            elif event.key == pygame.K_LEFT:
                vx -= THRUST
            elif event.key == pygame.K_RIGHT:
                vx += THRUST

    # Update the game state
    x += vx
    y += vy
    vy += GRAVITY

    # Check for collision with the ground
    if y > SCREEN_HEIGHT - 50:
        y = SCREEN_HEIGHT - 50
        vy = 0
        vx = 0
        score += 1
        x = SCREEN_WIDTH / 2
        y = SCREEN_HEIGHT / 2
        vx = 0
        vy = 0
        fuel = 100

    # Get the current game state as input for the Q learning model
    state = np.array([x, y, vx, vy])

    # Use the Q learning model to predict the action to take (0 = left, 1 = right)
    action = np.argmax(model.predict(state.reshape(1, 4)))

    # Take the action
    if action == 0:
        vx -= THRUST
    elif action == 1:
        vx += THRUST

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the lunar lander
    screen.blit(lander, (x, y))

    # Draw the fuel gauge
    pygame.draw.rect(screen, (255, 0, 0), (50, 50, fuel * 2, 25))

    # Draw the score
    font = pygame.font.Font(None, 36)
    text = font.render(str(score), 1, (255, 255, 255))
    screen.blit(text, (50, 25))

    # Update the display
    pygame.display.flip()

    # Limit the frame rate to 60 FPS
    clock.tick(60)

# Shut down Pygame
pygame.quit()
