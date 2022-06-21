#!/usr/bin/env python3
import pickle 
import numpy as np

# For plotting
import matplotlib.pyplot as plt

def plot_result(trial_data):

    if trial_data['x'] is not None and trial_data['r'] is not None:
        # Create a new figure
        fig = plt.figure()

        # Add a subplot to the figure
        ax = fig.add_subplot(111)
        ax.add_patch(plt.Rectangle((0.0, 0.0),
                        trial_data['l'], trial_data['l'],
                        fc ='none', 
                        ec ='black',
                        lw = 1))

        # Plot the circles inside the square
        _, num_circles = trial_data['x'].shape
        for i in range(num_circles):
            ax.add_patch(plt.Circle(trial_data['x'][:,i], trial_data['r'], ec = 'blue', fc='none'))

        # Set the limits of the plot
        plt.xlim([-1, trial_data['l']+1])
        plt.ylim([-1, trial_data['l']+1])
        ax.axis('equal')

        plt.show()

# The entrypoint of the program
def main():

    # ------------------------------------------
    # --- Circle Packing - 
    # ------------------------------------------
    datasets = []

    # Read the file of the small problem instance of circle packing
    open_file = open('small_problem_results.pkl', "rb")
    datasets.append(pickle.load(open_file))
    open_file.close()
    
    # Read the file of the large problem instance of circle packing
    open_file = open('large_problem_results.pkl', "rb")
    datasets.append(pickle.load(open_file))
    open_file.close()

    # Define the histogram parameters for each plot
    hist_params = []
    hist_params.append(
        # Parameters for the histogram of the small problem instance
        {'start_hist': 72, 'stop_hist': 80, 'step_hist': 0.25, 
        'start_xticks': 72, 'stop_xticks': 80, 'step_xticks':  1,
        'start_yticks':  0, 'stop_yticks': 80, 'step_yticks': 10})
    hist_params.append(
        # Parameters for the histogram of the large problem instance
        {'start_hist': 82.5, 'stop_hist': 88, 'step_hist': 0.125, 
        'start_xticks': 82.5, 'stop_xticks': 88, 'step_xticks':  0.5,
        'start_yticks':  0, 'stop_yticks': 60, 'step_yticks': 5})

    for j, data in enumerate(datasets):

        # List of percentage covered in each trial
        percentage_covered = []

        # Save the densest packing covered by the algorithm
        densest_packing = 0.0
        index_densest_packing = 0

        count_divergence = 0

        # Get the percentage covered
        for i, item in enumerate(data):

            # Check if the algorithm diverged due to numerical issues
            if (item['covered_area'] * 100.0 > 99.0):
                count_divergence += 1
                continue

            percentage_covered.append(item['covered_area'] * 100.0)

            # Update the densest packing
            if densest_packing < item['covered_area'] * 100.0:
                densest_packing = item['covered_area'] * 100.0
                index_densest_packing = i

        # Plot the histogram of the percentage covered
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.hist(percentage_covered, np.arange(start=hist_params[j]['start_hist'], stop=hist_params[j]['stop_hist'], step=hist_params[j]['step_hist']), fc='blue', ec='black')
        ax.set_xticks(np.arange(start=hist_params[j]['start_xticks'], stop=hist_params[j]['stop_xticks'], step=hist_params[j]['step_xticks']))
        ax.set_yticks(np.arange(start=hist_params[j]['start_yticks'], stop=hist_params[j]['stop_yticks'], step=hist_params[j]['step_yticks']))
        plt.xlabel('percentage covered')
        plt.ylabel('number of trials')
        plt.show()

        # Plot the result of the densest packing
        plot_result(data[index_densest_packing])

        # Compute the percentage of cases within 1% of the best packing
        count = 0
        for item in data:
            if densest_packing * 0.99 <= item['covered_area'] * 100:
                count = count + 1
        best_packing_percentage = count / len(data) * 100

        count = 0
        for item in data:
            if densest_packing * 0.98 <= item['covered_area'] * 100:
                count = count + 1
        best_packing_percentage_2 = count / len(data) * 100

        count = 0
        for item in data:
            if densest_packing * 0.97 <= item['covered_area'] * 100:
                count = count + 1
        best_packing_percentage_3 = count / len(data) * 100

        print('Densest Packing: ' + str(densest_packing))
        print('Cases within 1% of the best known packing: ' + str(best_packing_percentage) + '%')
        print('Cases within 2% of the best known packing: ' + str(best_packing_percentage_2) + '%')
        print('Cases within 3% of the best known packing: ' + str(best_packing_percentage_3) + '%')
        print('Radius of the circle in best packing: ' + str(data[index_densest_packing]['r']))
        print('Percentage of failure: ' + str(count_divergence / len(data) * 100.0) + '%')
        print('------------------------------------')

if __name__ == '__main__':
    main()