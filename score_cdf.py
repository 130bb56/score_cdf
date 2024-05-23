import matplotlib.pyplot as plt
import numpy as np

def parsing(file_path, column):
    # Column parsing corresponding to the arg:column
    with open(file_path, "r") as fd:
        data = [float(line.strip().split()[column]) for line in fd]
    return data

# Drawing the CDF graph
def draw(data):
    plt.figure(figsize=(10, 6))
    sorted_data = np.sort(data)

    # Y values represent the percentage of data points.
    y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
    plt.plot(sorted_data, y_values, marker='.', linestyle='none')

    # show y-axis as %
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    plt.title("Score cdf")
    plt.xlabel("Score")
    plt.ylabel("Cumulative Percentage")

    # Drawing graph
		plt.grid(True)
    plt.show()

###################### modify this area ######################
file_path = "data.txt"
data = parsing(file_path, 1)
##############################################################
draw(data)
