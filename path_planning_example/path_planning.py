from plot_paths import plot2D, plot3D
import cvxpy as cvx
import dccp
import numpy as np

# To install dccp extension for CVX
# pip3 install -U dccp

class PathPlanning:

    def __init__(self, dimension, number_samples, initial_pos, final_pos, obstacles_center, obstacles_radius):
        
        # Dimension of the problem
        self.d = dimension
        self.n = number_samples

        # Initial and final position
        self.a = initial_pos
        self.b = final_pos

        # Define the obstacles
        self.pj = obstacles_center
        self.rj = obstacles_radius
        self.m = obstacles_radius.size      # The number of obstacles

        # Define list of waypoints
        self.L = 0
        self.waypoints = []


    def solve(self):
        x = cvx.Variable((self.d, self.n+1))
        L = cvx.Variable()

        constr =  [x[:,0] == self.a]
        constr += [x[:,self.n] == self.b]
        
        for i in range(1, self.n+1):
            
            # Distance between any two consecutive points
            constr += [cvx.norm(x[:,i] - x[:,i-1]) <= L / self.n]
            
            # Not pass through obstacle contraint
            for j in range(self.m):
                constr += [cvx.norm(x[:,i] - self.pj[:,j]) >= self.rj[j]]
        
        # Solve the non-convex optimization problem using DCCP method
        problem = cvx.Problem(cvx.Minimize(L), constr)
        result = problem.solve(method='dccp', verbose=True)
        
        # Save the final list of waypoints
        self.waypoints = x.value

        # Save the total length of the curve
        self.L = L.value

def example_3d():

    # Define the initial position and final target position
    initial_pos = np.array([0.0, 0.0, 0.0])
    final_pos = np.array([15.0, 15.0, 15.0])

    # Define the obstacles
    obstacle_center = np.array([[3.0, 2.0, 2.0],
                                [5.0, 4.0, 6.0],
                                [10.0, 11.0, 10.0]]).T

    # Define the obstacle radius
    obstacle_radius = np.array([2, 2.0, 3.0])

    # Define the problem
    problem = PathPlanning(3, 50, initial_pos, final_pos, obstacle_center, obstacle_radius)
    problem.solve()

    # Plot the trajectory in the obstacle map
    plot3D(obstacle_center, obstacle_radius, problem.waypoints)
    print("Path length: " + str(problem.L))

    # Define the initial position and final target position
    initial_pos = np.array([0.0, 0.0, 0.0])
    final_pos = np.array([15.0, 15.0, 15.0])

    # Define the obstacles
    obstacle_center = np.array([[3.0, 2.0, 2.0],
                                [5.0, 4.0, 6.0],
                                [10.0, 11.0, 10.0]]).T

def example_2d():

    # Define the initial position and final target position
    initial_pos = np.array([0.0, 0.0])
    final_pos = np.array([15.0, 15.0])

    # Define the obstacles
    obstacle_center = np.array([[3.0, 2.0],
                                [6.0, 6.0],
                                [10.0, 11.0]]).T

    # Define the obstacle radius
    obstacle_radius = np.array([2.0, 2.0, 3.0])

    # Define the problem
    problem = PathPlanning(2, 50, initial_pos, final_pos, obstacle_center, obstacle_radius)
    problem.solve()

    # Plot the trajectory in the obstacle map
    plot2D(obstacle_center, obstacle_radius, problem.waypoints)
    print("Path length: " + str(problem.L))

if __name__ == "__main__":

    # ---------------- Example in 3D -----------------------
    example_3d()

    # ---------------- Example in 2D -----------------------
    example_2d()