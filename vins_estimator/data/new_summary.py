import os
import pandas as pd
import numpy as np
dir_list = ["exp_1","exp_2","exp_3","exp_4","exp_5"]
euclidean_dist = {key: [] for key in dir_list}

def form_statement(x,y,z,roll,pitch,yaw, j, dir):
    prefix = "original"
    if(j == "1"):
        prefix = "with alignment"
    elif(j == "0"):
        prefix = "without alignment"

    euclidean_dist[dir].append(pow(x*x+y*y+z*z,0.5))
    return prefix + " & " + "{:.3f}".format((round(x,3))) + " & " + "{:.3f}".format((round(y,3))) + " & "  + "{:.3f}".format((round(z,3))) + " & " + "{:.3f}".format((round(roll,3))) + " & " + "{:.3f}".format((round(pitch,3))) + " & " + "{:.3f}".format((round(yaw,3))) + " \\\\"

def analysis(file1,file2, dir, j="2"):
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
    print("\\hline")
    print(form_statement(avg_table['x'],avg_table['y'],avg_table['z'],avg_table['roll'],avg_table['pitch'],avg_table['yaw'],j, dir))

def process(dir):
    print()
    file1_dir = os.path.join(dir,"data1.txt")
    file2_dir = os.path.join(dir,"data2.txt") 
    file1 = pd.read_csv(file1_dir, delimiter=",")
    file2 = pd.read_csv(file2_dir, delimiter=",") 
    analysis(file1, file2, dir, "2")
    for j in ["0","1"]:
        file1_dir = os.path.join(dir,"data1_6.000000_" + j + ".txt")
        file2_dir = os.path.join(dir,"data2_6.000000_" + j + ".txt")
        file1 = pd.read_csv(file1_dir, delimiter=",")
        file2 = pd.read_csv(file2_dir, delimiter=",")
        analysis(file1, file2, dir, j)

for dir in dir_list:
    process(dir)

# print(euclidean_dist)
for idx,j in enumerate(["original", "no alignment", "alignment"]):
    print("\\hline")
    tmp = j + " & "
    for key, val in euclidean_dist.items():
        tmp += "{:.3f}".format((round(euclidean_dist[key][idx],3))) + " & "

    print(tmp)
    
print("\\hline")
