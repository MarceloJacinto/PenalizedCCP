import numpy as np
import matplotlib as mtl
import matplotlib.pyplot as plt

def plot3D(obstacles_center, obstacles_radius, x):
    

    # Check if the dimensions of the centers and radius are the same
    assert obstacles_radius.size == obstacles_center.size / 3
    
    # Create a 3D figure
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Plot the vehicle path
    line_path, = ax.plot(x[0,:], x[1,:], x[2,:], 'red', label='Path')

    # Create 3D spheres for the obstacles to plot
    for i in range(obstacles_radius.size):
        
        # Create a meshgrid for the plot
        r = obstacles_radius[i]
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]

        x = obstacles_center[0, i] + r * np.cos(u) * np.sin(v)
        y = obstacles_center[1, i] + r * np.sin(u) * np.sin(v)
        z = obstacles_center[2, i] + r * np.cos(v)
        ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r)

    fake2Dline = mtl.lines.Line2D([0],[0], linestyle="none", c='b', marker = 'o')

    ax.legend([line_path, fake2Dline], ['Path', 'Obstacles'])

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.invert_xaxis()

    plt.show()

def plot2D(obstacles_center, obstacles_radius, x):
    

    # Check if the dimensions of the centers and radius are the same
    assert obstacles_radius.size == obstacles_center.size / 2
    
    # Create a 3D figure
    fig = plt.figure()
    ax = plt.axes()

    # Plot the vehicle path
    line_path, = ax.plot(x[0,:], x[1,:], 'red', label='Path')

    # Create 3D spheres for the obstacles to plot
    for i in range(obstacles_radius.size):
        
        # Create a meshgrid for the plot
        circle = plt.Circle(obstacles_center[:,i], obstacles_radius[i], color='blue')
        ax.add_patch(circle)
    
    ax.legend([line_path, circle], ['Path', 'Obstacles'])

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.show()