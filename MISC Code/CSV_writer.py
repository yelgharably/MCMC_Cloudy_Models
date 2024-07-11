import numpy as np
import os
import pandas as pd

Ap = np.array([
    'ApA', 'ApA', 'ApA', 'ApA'
])

lines = np.array(['HI_1216','Blnd_1240'Blnd_1549
HeII_1640
])

rfluxes_measured = np.array([
    0.20, 4.12, 2.80, 0.34, 0.91, 0.12, 2.55, 100.0, 7.59, 1.21, 7.33, 6.49, 
    1.25, 2.63, 1.21, 0.95, 0.22, 0.17, 0.64, 0.20, 0.73, 0.13
])

rfluxes_lower = np.array([
    0.15, 3.74, 2.36, 0.29, 0.77, 0.09, 1.86, 99.9, 5.54, 1.09, 5.55, 4.94, 
    1.07, 2.22, 1.10, 0.88, 0.15, 0.11, 0.57, 0.13, 0.64, 0.09
])

rfluxes_upper = np.array([
    0.25, 4.48, 3.08, 0.41, 1.01, 0.21, 3.02, 100.1, 8.99, 1.34, 7.76, 6.93, 
    1.51, 2.86, 1.38, 1.12, 0.30, 0.24, 0.72, 0.27, 0.87, 0.18
])

rfluxes_err_up = rfluxes_upper - rfluxes_measured
rfluxes_err_low = rfluxes_measured - rfluxes_lower
rfluxes_err = (rfluxes_err_up + rfluxes_err_low) / 2

lines_vac = np.array([
    5756.24, 5877.27, 6302.04, 6313.85, 6365.54, 6373.12, 6549.86, 6564.62, 6585.27, 
    6679.99, 6718.29, 6732.68, 7076.23, 7137.75, 7321.94, 7332.21, 7379.86, 7413.65, 
    7753.23, 8440.28, 8448.57, 8469.58
])

combined = np.column_stack((lines, rfluxes_measured, rfluxes_err, lines_vac))

data = pd.DataFrame(combined, columns=['lines', 'rfluxes_measured', 'rfluxes_err', 'lines_vac'])

# Change directory accordingly
directory_path = r"D:\Undergraduate Life\Summer 2024\Trainor_Research"
file_name = 'lines.csv'
file_path = os.path.join(directory_path, file_name)
data.to_csv(file_path, index=False) #Index could be changed to true if you need an extra column of indices
