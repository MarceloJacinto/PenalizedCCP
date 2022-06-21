#!/usr/bin/env python3

# For mathematical computations
from typing import Union, List
import pickle 
import numpy as np
import cvxpy as cvx

# For plotting
import matplotlib.pyplot as plt

# For running multiple operations in parallel
from joblib import Parallel, delayed


class CirclePackingSolver:

    def __init__(self,
        l: float,
        n: float, 
        x_0: np.ndarray, 
        mu: float, 
        tau_0: float, 
        tau_max: float,
        delta: float):
        """
        Class constructor - initializes all common variables, and solver configurations
        """

        # Make sure the mu is greater than 1 and tau_0 is greater than 0
        assert mu > 1
        assert tau_0 > 0

        # Data to solve the problem
        self.l: float = l                   # The side-length of the square
        self.n: float = n                   # The number of circles

        # Make sure that the initial conditions are of the form (2, n)
        assert x_0.shape == (2, n)

        # Initial conditions for the problem
        self.x_0: np.ndarray = x_0            # The initial position of each circle inside the square

        # Solver constants
        self.mu: float = mu
        self.tau_0: float = tau_0
        self.tau_max: float = tau_max

        # Initialize the iteration counter 
        self.k: int = 0

        # Stopping criteria
        self.delta = delta

        # Final result variables
        self.x: np.ndarray = None
        self.r: float = 0.0

        # Final slack variables
        self.s1: Union[cvx.Variable, np.ndarray] = None
        self.s2: Union[cvx.Variable, np.ndarray] = None
        self.s3: Union[cvx.Variable, np.ndarray] = None

        # Performance metrics
        self.stop_function_value: float = None
        self.iteration_time: List[float] = []
        self.cost: List[float] = []

    def solve(self, enforce_boundary_constraints:bool = False, verbose: bool = True):
        """
        Method that actually solves the optimization problem. If 'enforce_boundary_constraints' is set to True, then
        we will enforce boundary constraints without slack variables on the problem. Additionally, the remaining constraints
        will be enforced with slack variables using a cutting plane method, that is, at each iteration, the 22n constraints with
        the smallest margin are considered or all currently violated constraints (whichever set is larger)
        """

        # Initialize the auxiliar variables
        stop = False
        self.tau_k = self.tau_0
        x_k = self.x_0

        # --- Declare the optimization variables --- 
        x = cvx.Variable(x_k.shape)
        r = cvx.Variable(1)

        # Iterate until stopping criteria is met
        while not stop:
            print('Iterations Completed: ' + str(self.k))
            
            # If we don't to enforce boundary constraints, we want to use slack variables to all inequalities
            if not enforce_boundary_constraints:
                s1 = cvx.Variable(int(self.n * (self.n-1) / 2))
                s2 = cvx.Variable((2,self.n))
                s3 = cvx.Variable((2,self.n))

                # Cost function when we do not enforce boundary constraints nor apply the cutting plane method
                cost_function = -r + (self.tau_k * cvx.sum(s1)) + (self.tau_k * cvx.sum(s2)) + (self.tau_k * cvx.sum(s3))

                # Define the constraints list
                constr = self.set_of_constraints_1(r, x, x_k, s1) + self.set_of_constraints_2_and_3(r, x, x_k, s2, s3)

                # If we are using slack variables for all the constraints (including boundary constraints), they must be greater than 0
                constr += self.set_relaxation_contraints(s1, s2, s3)

            # Algorithm variation where we not only enforce boundary constraints, but also use cutting plane to select the other constraints
            else:
                constr, s1 = self.cutting_plane_constraints_1(r, x, self.r, x_k)
                s2 = np.zeros((2,self.n))
                s3 = np.zeros((2,self.n))

                # Cost function when we enforce boundary constraints and apply the cutting plane method
                cost_function = -r + (self.tau_k * cvx.sum(s1))

                # Add to the cutting plane constraints define previously the 
                constr += self.set_of_constraints_2_and_3(r, x, x_k, s2, s3)
                # Otherwise, we are just using slack variables for the remaining constraints and they must be greater than 0
                constr += self.set_relaxation_contraints_variation(s1)


            # --- Solve the optimization problem --- 
            problem = cvx.Problem(cvx.Minimize(cost_function), constr)
            optimal_value = problem.solve(verbose=verbose)

            # --- Update the problem statistics ---
            self.iteration_time.append(problem.solver_stats.solve_time)
            self.cost.append(optimal_value)

            # return the final result and the number of iterations
            self.r = float(r.value)
            self.x = x.value
            self.s1 = s1
            self.s2 = s2
            self.s3 = s3

            # --- Update the iteration and check the stoping criteria --- 
            self.k = self.k + 1

            # --- Check if we met the stopping criteria --- 
            stop = self.stoping_criteria(self.r, s1.value, s2.value if not enforce_boundary_constraints else s2, s3.value if not enforce_boundary_constraints else s3)

            # --- Update tau --- 
            self.tau_k = np.min([self.mu * self.tau_k, self.tau_max])

            # --- Update the variable x_k ---
            x_k = x.value

    def stoping_criteria(self, r_k: float, s1_k: np.ndarray, s2_k: np.ndarray, s3_k:np.ndarray) -> bool:

        stop: bool = False

        # Check if we have already reached the maximum tau value
        if self.tau_k == self.tau_max:
            stop = True

        # Check if this is the first iteration (we cannot computer the stop condition expression yet)
        if self.k != 1:
            # Stop condition: (f0(xk) - g0(xk) + tau_k * sum(sk)) - (f0(xk+1) - g0(xk+1) + tau_k * sum(sk+1)) <= delta
            # i.e. the improvement in the objective is small, and bellow a given threshold
            self.stop_function_value =  (0.0 - self.prev_rk + (self.tau_k * (np.sum(self.prev_s1_k) + np.sum(self.prev_s2_k) + np.sum(self.prev_s3_k)))) - (0.0 - r_k + (self.tau_k * (np.sum(s1_k) + np.sum(s2_k) + np.sum(s3_k))))

            if self.stop_function_value <= self.delta: 
                stop = True

        # Save the r_k, s1_k, s2_k, s3_k for the next iteration
        self.prev_rk = r_k
        self.prev_s1_k = s1_k
        self.prev_s2_k = s2_k
        self.prev_s3_k = s3_k

        # Return the stop condition value
        return stop

    def set_of_constraints_1(self, r: cvx.Variable, x: cvx.Variable, x_k: np.ndarray, s1: cvx.Variable) -> List[cvx.constraints.nonpos.Inequality]:
        """
        Constraints for the regular (non-intersecting circles) without using the cutting plane method
        """
        
        # Create an empty list of constraints
        constr = []

        # Iterate over i and j
        k = 0
        for i in range(0, self.n - 1):
            for j in range(i+1, self.n):

                # Compute fij
                fij = 4 * cvx.power(r, 2)

                # Compute gij_hat
                gij_hat = np.power(np.linalg.norm(x_k[:,i] - x_k[:,j]), 2) \
                    + (2 * (x_k[:,i] - x_k[:,j]).T @ (x[:,i] - x_k[:,i])) \
                    + (2 * (x_k[:,j] - x_k[:,i]).T @ (x[:,j] - x_k[:,j]))

                # Construct the constraint
                constr.append(fij - gij_hat <= s1[k])

                # Increment k
                k = k + 1
        
        return constr

    def cutting_plane_constraints_1(self, r: float, x: np.ndarray, r_k: np.ndarray, x_k: np.ndarray):
        """
        Constraints for non-intersecting circles, using the cutting plane method
        """

        # Create an empty list of constraints
        set_of_violated_constraints = []
        set_of_22n_constraints = []

        # Iterate over i and j
        k = 0
        for i in range(0, self.n - 1):
            for j in range(i+1, self.n):
                
                # Compute the actual fij using the current radius of the circles
                fij = 4 * np.power(r_k, 2)
                
                # Compute the actual gij using the current radius and points
                gij = np.power(np.linalg.norm(x_k[:,i] - x_k[:,j]), 2)

                # Add the value to the list of all constraints
                set_of_22n_constraints.append((i, j, fij-gij))

                # Add the constraint to the list if the inequality is violated
                if fij - gij > 0:
                    set_of_violated_constraints.append((i, j, fij - gij))

        # Restricting the set to the 22n largest constraints
        set_of_22n_constraints = sorted(set_of_22n_constraints, key=lambda x: x[2], reverse=True)
        set_of_22n_constraints = set_of_22n_constraints[:self.n*22]

        # Choose the set with the largest number of constraints
        set_of_constraints = set_of_22n_constraints if len(set_of_22n_constraints) >= len(set_of_violated_constraints) else set_of_violated_constraints

        # Generate the constraints with slack variables
        s1 = cvx.Variable(len(set_of_constraints))

        # Compute the inequallity constraints 
        constr = []
        for k, item in enumerate(set_of_constraints):
            
            # Get the indexes of the points
            i = item[0]
            j = item[1]

            # Compute fij
            fij = 4 * cvx.power(r, 2)

            # Compute gij_hat
            gij_hat = np.power(np.linalg.norm(x_k[:,i] - x_k[:,j]), 2) \
                + (2 * (x_k[:,i] - x_k[:,j]).T @ (x[:,i] - x_k[:,i])) \
                + (2 * (x_k[:,j] - x_k[:,i]).T @ (x[:,j] - x_k[:,j]))

            # Construct the constraint
            constr.append(fij - gij_hat <= s1[k])

        print(len(constr))

        return constr, s1


    def set_of_constraints_2_and_3(self, r: cvx.Variable, x: cvx.Variable, x_k: np.ndarray, s2:  Union[cvx.Variable, np.ndarray], s3: Union[cvx.Variable, np.ndarray]) -> List[cvx.constraints.nonpos.Inequality]:
        """
        Setting the boundary constraints
        """

        # Create an empty list of constraints
        constr_2 = []
        constr_3 = []

        # Define the constants to use in all inequalities
        constant_2 = np.ones((2, 1)) @ (self.l - r)
        constant_3 = np.ones((2,1)) @ r

        for i in range(self.n):
            constr_2.append(x[:,i] - constant_2 <= s2[:,i])
            constr_3.append(constant_3 - (2 * x[:,i] - x_k[:,i]) <= s3[:,i])

        return constr_2 + constr_3

    def set_relaxation_contraints(self, s1: cvx.Variable, s2: cvx.Variable, s3: cvx.Variable) -> List[cvx.constraints.nonpos.Inequality]:
        """
        Setting the contraints for the slack variables in the regular scenario where all the contraints have an associated slack variable
        that must be greater or equal than 0.0
        """

        # Create an empty list of contraints
        constr = []

        # Make all s1 greater than 0.0
        for i in range(s1.size):
            constr.append(s1[i] >= 0.0)

        # Make all s2 entries greater than 0.0
        rows, cols = s2.shape
        for i in range(rows):
            for j in range(cols):
                constr.append(s2[i, j] >= 0.0)

        # Make all s2 entries greater than 0.0
        rows, cols = s3.shape
        for i in range(rows):
            for j in range(cols):
                constr.append(s3[i, j] >= 0.0)

        return constr

    def set_relaxation_contraints_variation(self, s1):
        """
        Setting the constraints for the slack variables associated with non-intersecting circles constraints. This method is only used
        when we are in the scenario where we enforce boundary constraints without slack variables
        """
        # Create an empty list of contraints
        constr = []

        # Make all s1 greater than 0.0
        for i in range(s1.size):
            constr.append(s1[i] >= 0.0)
        
        return constr

    def plot_result(self):

        if self.x is not None and self.r is not None:
            # Create a new figure
            fig = plt.figure()

            # Add a subplot to the figure
            ax = fig.add_subplot(111)
            ax.add_patch(plt.Rectangle((0.0, 0.0),
                            self.l, self.l,
                            fc ='none', 
                            ec ='black',
                            lw = 1))

            # Plot the circles inside the square
            _, num_circles = self.x.shape
            for i in range(num_circles):
                ax.add_patch(plt.Circle(self.x[:,i], self.r, ec = 'blue', fc='none'))

            # Set the limits of the plot
            plt.xlim([-1, self.l+1])
            plt.ylim([-1, self.l+1])
            ax.axis('equal')

            plt.show()

    def get_problem_stats(self):
        return {
            'l': self.l, 'n': self.n, 
            'delta': self.delta, 'tau_max': self.tau_max, 
            'tau_0': self.tau_0, 'mu': self.mu, 
            'x': self.x, 'r': self.r, 
            's1': self.s1, 's2': self.s2, 's3': self.s3,
            'k': self.k,
            'tau_k': self.tau_k,
            'stop_condition': self.stop_function_value,
            'iteration_time': self.iteration_time,
            'cost': self.cost,
            'covered_area': self.covered_area()}

    def covered_area(self) -> float:

        # Compute the percentage of covered area 
        if self.r is not None and self.x is not None:
            
            # Get the number of circles
            _, num_circles = self.x.shape

            # Compute the area ocupied by the circles
            occupied_area = num_circles * np.pi * np.power(self.r, 2)

            # Compute the total area
            total_area = self.l * self.l

            # Return the occupied percentage
            return occupied_area / total_area
        
        return 0.0
  

def small_problem_instance(problem_number: int):

    print('Problem no. ' + str(problem_number))

    l = 10.0        # The side length of the box
    n = 41          # The number of circles

    # Defining the solver constants
    tau_0 = 1.0
    mu = 1.5
    tau_max = 10000

    # Defining the stopping criterium value
    delta = 1E-16

    # Draw x_0 from the uniform distribution [0, l] x [0, l]
    x_0 = np.random.uniform(low=0, high=l, size=(2, n))
    
    # Initiate the circle packing problem
    circle_packing = CirclePackingSolver(l, n, x_0, mu, tau_0, tau_max, delta)

    # Try to solve the problem, but don't freak out if cannot find a solution
    try:
        circle_packing.solve(enforce_boundary_constraints=False, verbose=False)
    except Exception as e:
        pass

    # Return the problem stats
    return circle_packing.get_problem_stats()

def large_problem_instance(problem_number: int):

    print('Problem no. ' + str(problem_number))

    l = 10.0        # The side length of the box
    n = 400         # The number of circles

    # Defining the solver constants
    tau_0 = 0.001
    mu = 1.05
    tau_max = 10000

    # Defining the stopping criterium value
    delta = 1E-16

    # Draw x_0 from the uniform distribution [0, l] x [0, l]
    x_0 = np.random.uniform(low=0, high=l, size=(2, n))
    
    # Initiate the circle packing problem
    circle_packing = CirclePackingSolver(l, n, x_0, mu, tau_0, tau_max, delta)

    # Try to solve the problem, but don't freak out if cannot find a solution
    try:
        circle_packing.solve(enforce_boundary_constraints=True, verbose=False)
    except Exception as e:
        pass

    # Return the problem stats
    return circle_packing.get_problem_stats()

def demo1():
    l = 10.0        # The side length of the box
    n = 41          # The number of circles

    # Defining the solver constants
    tau_0 = 1.0
    mu = 1.5
    tau_max = 10000

    # Defining the stopping criterium value
    delta = 1E-16

    # Draw x_0 from the uniform distribution [0, l] x [0, l]
    x_0 = np.random.uniform(low=0, high=l, size=(2, n))
    
    # Initiate the circle packing problem
    circle_packing = CirclePackingSolver(l, n, x_0, mu, tau_0, tau_max, delta)

    # Solve the problem
    circle_packing.solve(enforce_boundary_constraints=True, verbose=False)

    # Compute the covered area
    percentage_covered = circle_packing.covered_area()
    print('Percentage of covered area: ' + str(percentage_covered))

    # Get optimization stats
    stats = circle_packing.get_problem_stats()
    print('Radius of the circles: ' + str(stats['r']))

    # Plot the final result
    circle_packing.plot_result()

def demo2():
    
    l = 10.0        # The side length of the box
    n = 400         # The number of circles

    # Defining the solver constants
    tau_0 = 0.001
    mu = 1.05
    tau_max = 10000

    # Defining the stopping criterium value
    delta = 1E-16

    # Draw x_0 from the uniform distribution [0, l] x [0, l]
    x_0 = np.random.uniform(low=0, high=l, size=(2, n))
    
    # Initiate the circle packing problem
    circle_packing = CirclePackingSolver(l, n, x_0, mu, tau_0, tau_max, delta)

    # Solve the problem
    circle_packing.solve(enforce_boundary_constraints=True, verbose=True)

    # Compute the covered area
    percentage_covered = circle_packing.covered_area()
    print('Percentage of covered area: ' + str(percentage_covered))

    # Get optimization stats
    stats = circle_packing.get_problem_stats()
    print('Radius of the circles: ' + str(stats['r']))

    # Plot the final result
    circle_packing.plot_result()


# Actual function used to generate the "large_problem_results.pkl" and "small_problem_results.pkl"
def main():

    # Compute the results for the small instance problem
    results = Parallel(n_jobs=2)(delayed(small_problem_instance)(i) for i in range(1000))

    # Save the results to a pickle file
    open_file = open('small_problem_results.pkl', "wb")
    pickle.dump(results, open_file)
    open_file.close()

    # Compute the results for the large instance problem
    results = Parallel(n_jobs=2)(delayed(large_problem_instance)(i) for i in range(450))

    # Save the results to a pickle file
    open_file = open('large_problem_results.pkl', "wb")
    pickle.dump(results, open_file)
    open_file.close()

if __name__ == '__main__':
    main()