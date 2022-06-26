#!/usr/bin/env python3

"""
MIT License

Copyright (c) 2022 Marcelo Jacinto

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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
