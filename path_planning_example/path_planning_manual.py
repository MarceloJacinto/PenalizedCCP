from plot_paths import plot2D, plot3D
import numpy as np
import cvxpy as cvx

class PathPlanning:

    def __init__(self, dimension, number_samples, initial_pos: np.ndarray, final_pos: np.ndarray, obstacles_center: np.ndarray, obstacles_radius: np.ndarray, tau_0: float, tau_inc: float, tau_max: float, delta: float):
        
        # Dimension of the problem
        self.d = dimension
        self.n = number_samples

        # Define the parameters for the vehicle
        self.a = initial_pos      # The vehicle's initial position
        self.b = final_pos        # The target endpoint
        
        # Define the obstacles
        self.pj = obstacles_center
        self.rj = obstacles_radius
        self.m = obstacles_radius.size      # The number of obstacles

        assert obstacles_radius.size == obstacles_center.size / dimension

        # Parameters to solve the optimization problem
        self.tau_0 = tau_0          # The line search adjustment
        self.tau_inc = tau_inc
        self.tau_max = tau_max
        self.delta = delta          # The stopping criteria value (improvement in objective function)

        # Define list of waypoints (solution to the problem)
        self.L = 0
        self.waypoints = []

    def solve(self):
        
        # Setup the optimization variables
        L = cvx.Variable(shape=1)
        X = cvx.Variable(shape=(self.d, self.n))

        # Setup the slack variables
        s = cvx.Variable(shape=(self.n * self.m,))

        # Setup the Xk variable as a parameter (that is changing over each iteration - to avoid compilations overhead)
        Xk = cvx.Parameter(shape=(self.d, self.n))
        tau_k = cvx.Parameter()
        tau_k.value = self.tau_0

        # Initialize Xk with a line between 2 points
        aux = np.zeros((self.d, self.n))
        for i in range(self.d):
            aux[i,:] = np.linspace(self.a[i], self.b[i], num=self.n)
        Xk.value = aux

        # Setup the cost function
        cost_function = L + (tau_k * cvx.sum(s))

        # Setup the constraints
        constr  = [X[:,0] - self.a == 0]
        constr += [X[:,self.n-1] - self.b == 0]
        constr += [cvx.norm(X[:,i] - X[:,i-1]) - ((1.0 / self.n) * L) <= 0.0 for i in range(1, self.n)]
        constr += [self.rj[j] - cvx.norm(Xk[:,i] - self.pj[:,j]) - ((Xk[:,i] - self.pj[:,j]) @ (X[:,i] - Xk[:,i]) / cvx.norm(Xk[:,i] - self.pj[:,j])) <= s[j + (i * self.m)] for i in range(0, self.n) for j in range(0, self.m)]
        constr += [s >= np.zeros(self.n * self.m)]

        # Compile the problem before-hand
        problem = cvx.Problem(cvx.Minimize(cost_function), constraints=constr)
        
        stop: bool = False         # Setup the stop flags
        iteration_count: int = 0

        # Initialize the Path length variable
        prev_L: float = 0.0
        prev_s: np.ndarray = np.zeros(self.n * self.m)
        prev_tau: float = tau_k.value

        while not stop:

            # Solve the problem
            problem.solve(verbose=True, warm_start=True)

            # Update the tau_k value
            prev_tau = tau_k.value
            tau_k.value = np.minimum(self.tau_inc * tau_k.value, self.tau_max)

            # Update the number of iterations
            iteration_count += 1
            print('Iteration: ' + str(iteration_count))

            # Check the stopping criteria
            if iteration_count > 1:
                stop = self.stopping_criteria(prev_L, L.value, prev_tau, tau_k.value, prev_s, s.value)

            # Update the values of the variables
            prev_L: float = L.value[0]
            prev_s: np.ndarray = s.value

            # Update the values of the points
            Xk.value = X.value
        
        # Save the final list of waypoints
        self.waypoints = X.value

        # Save the total length of the curve
        self.L = L.value[0]


    def stopping_criteria(self, prev_L, next_L, prev_tau_k, next_tau_k, prev_s, next_s) -> bool:
        
        # Last iteration objective function value
        prev_obj = prev_L + prev_tau_k * np.sum(prev_s)

        # Current objective function value
        curr_obj = next_L + next_tau_k * np.sum(next_s)

        return prev_obj - curr_obj <= self.delta

def example_2d():

    # Define the solver parameters
    tau_0 = 1.0
    tau_inc = 1.5
    tau_max = 10000
    delta = 1E-16

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
    problem = PathPlanning(2, 50, initial_pos, final_pos, obstacle_center, obstacle_radius, tau_0, tau_inc, tau_max, delta)
    problem.solve()

    # Plot the trajectory in the obstacle map
    plot2D(obstacle_center, obstacle_radius, problem.waypoints)
    print("Path length: " + str(problem.L))

def example_3d():

    # Define the solver parameters
    tau_0 = 1.0
    tau_inc = 1.5
    tau_max = 10000
    delta = 1E-16

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
    problem = PathPlanning(3, 50, initial_pos, final_pos, obstacle_center, obstacle_radius, tau_0, tau_inc, tau_max, delta)
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

if __name__ == '__main__':
    
    # ---------------- Example in 3D -----------------------
    example_3d()

    # ---------------- Example in 2D -----------------------
    example_2d()
    