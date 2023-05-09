import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt

zs = []
ws = []
es = []

def form_statement(x,y,z, weight):
    euclidean = pow(x*x+y*y+z*z,0.5)
    if(euclidean < 50):
        ws.append(weight)
        zs.append(z)
        es.append(euclidean)
        return  "{:.1f}".format((round(weight,1))) + "& " + "{:.3f}".format((round(euclidean,3))) + " \\\\"
    else:
        return ""

def get_summary(dir, dir_GT, weight):
    file1 = pd.read_csv(dir, delimiter=",")
    file2 = pd.read_csv(dir_GT, delimiter=",")
    # Get the breakpoint index
    breakpoint_value = 100 
    breakpoint_index = 0
    for idx,t in enumerate(file1.iloc[:, 0].tolist()):
        if(t > breakpoint_value):
            breakpoint_index = idx
            break
    # Remove all rows above the breakpoint index
    file1 = file1.iloc[breakpoint_index:]
    file2 = file2.iloc[breakpoint_index:]
    file1.columns = ['t', 'x', 'y','z','roll','pitch','yaw']
    file2.columns = ['t', 'x', 'y','z','roll','pitch','yaw']
    # Get the difference between each cell in the tables
    diff_table = file1 - file2
    angles_diff = np.abs(file1.iloc[:, -3:] - file2.iloc[:, -3:])
    angles_diff = np.where(angles_diff <= np.pi/2, angles_diff, np.pi - angles_diff)
    diff_table.iloc[:, -3:] = angles_diff
    diff_table = abs(diff_table)

    # Get the average for each column in the difference table
    avg_table = diff_table.mean()

    print(form_statement(avg_table['x'],avg_table['y'],avg_table['z'],weight))
    print("\\hline")

folder = "weight_optimization_exp2"
files = os.listdir(folder)

# Define a regular expression to extract the numeric value from each filename
pattern = re.compile(r'(\d+\.\d+)_\.txt$')

# Define a key function that extracts the numeric value from each filename using the regular expression
def sort_key(filename):
    match = pattern.search(filename)
    if match:
        return float(match.group(1))
    else:
        return filename
# Sort the list of filenames using the key function
sorted_files = sorted(files, key=sort_key)

print("\\hline")
print("Weight & Euclidean Error(m) \\\\")
print("\\hline")
for file in sorted_files:
    f = os.path.join(folder, file)
    if(file.startswith("data1")):
        f_split = file.split("_")
        weight = float(f_split[1])
        f_GT = os.path.join(folder, "data2_" + ("_").join(f_split[1:-1]) + "_.txt")
        get_summary(f,f_GT, weight)


# create a figure with two subplots
fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

# plot the graph of zs vs. ws in the first subplot
# ax1.plot(ws, zs)
# ax1.scatter(ws, zs, color='b')
# ax1.set_xlabel('Weightage')
# ax1.set_ylabel('Euclidean Error')

# plot the graph of es vs. ws in the second subplot
ax1.plot(ws, es)
ax1.scatter(ws, es, color='b')
ax1.set_xlabel('Weightage')
ax1.set_ylabel('Euclidean Error')

# Use enumerate to keep track of the indexes
indexed_list = list(enumerate(es))

# Sort the indexed list by value and get the first three indexes
min_three_indexes = [index for index, value in sorted(indexed_list, key=lambda x: x[1])][:3]

min_three_values = [ws[index] for index in min_three_indexes]

print(min_three_values)

# display the plots
plt.show()