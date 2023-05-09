import pandas as pd
import numpy as np
from scipy.linalg import logm
import cv2

def form_statement(x,y,z,roll,pitch,yaw):
    return "& " + "{:.3f}".format((round(x,3))) + " & " + "{:.3f}".format((round(y,3))) + " & "  + "{:.3f}".format((round(z,3))) + " & " + "{:.3f}".format((round(roll,3))) + " & " + "{:.3f}".format((round(pitch,3))) + " & " + "{:.3f}".format((round(yaw,3))) + " \\\\"

# Read the two input files
file1 = pd.read_csv("02/no_allign_data1.txt", delimiter=",")
file2 = pd.read_csv("02/no_allign_data2.txt", delimiter=",")
# file1 = pd.read_csv("allign_data1.txt", delimiter=",")
# file2 = pd.read_csv("allign_data2.txt", delimiter=",")
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

print(form_statement(avg_table['x'],avg_table['y'],avg_table['z'],avg_table['roll'],avg_table['pitch'],avg_table['yaw']))

roll = avg_table['roll']
pitch = avg_table['pitch']
yaw = avg_table['yaw']
rotation_matrix, _ = cv2.Rodrigues(np.array([roll, pitch, yaw]))
angle_axis, _ = cv2.Rodrigues(rotation_matrix)
print(angle_axis)