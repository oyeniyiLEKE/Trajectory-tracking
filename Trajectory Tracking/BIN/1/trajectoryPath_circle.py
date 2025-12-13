
"""
Differential Drive
Mobile Robot

This coordinatessimulation
Kinematics problem

"""


import pygame
import numpy as np

#initialize pygame
pygame.init()

#set the window size
size = width, height = 1000,800
screen = pygame.display.set_mode(size)
#clock
clock = pygame.time.Clock()
#Load robot image
image = pygame.image.load('robot.png').convert_alpha()

# Adjust the robot image size
robot_image_size = (150, 150)
#scale image to desired size
image = pygame.transform.scale(image, robot_image_size)


# this is a frame counter
i = 0
#load the simullation data that is computed by solving the direct problem

#solutionArray = np.load('simulationData_circle_rk4.npy')
solutionArray = np.load('SimData_circle_rk4.npy')
solutionArray = solutionArray

# x coordinate of the center of the robot
x = solutionArray[:,0]
# y coordinate of the center of the robot
y = solutionArray[:,1]


# Rotation angle of the robot, convert from radians to degrees
angle = -1*(360/(2*np.pi))*solutionArray[:,2]


#the list is used to store tuples of coordinates for tracking the robot trajectory
coordinates = []
#trajectory color, RGB color code
color_trajectory = (000, 000, 255)      

while (i<len(x)):
    # close window event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = True
       
    # RGB Background color       
    screen.fill((255, 255, 255))
    
    
    # Rotate robot image according to orientation
    image1 = pygame.transform.rotate(image, angle[i])
     
    image1_rect = image1.get_rect(center=(x[i], y[i]))
    
    
    screen.blit(image1, image1_rect)
    
    coordinates.append((x[i], y[i]))
    
    if i > 1:
        pygame.draw.lines(screen, color_trajectory,
                          False, coordinates, 3)
        
            
    pygame.display.flip()
    
    pygame.time.delay(1)
    
    clock.tick(1000) #pauses just enough to make sure the loop runs at 100 FPS (frames per second).
    i = i+1
        
        
pygame.quit()