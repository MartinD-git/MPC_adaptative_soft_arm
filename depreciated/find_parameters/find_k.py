import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#import the file
df = pd.read_csv('find_parameters/FEM data helicoid_D40.csv')

fig, ax = plt.subplots(2,1)
ax[0].plot(df['rad_X'], df['N*mm_X'],'r-', label='BendingX')
ax[0].plot(df['rad_Y'], df['N*mm_Y'],'b-', label='BendingY')
ax[0].set_xlabel('Angle [rad]')
ax[0].set_ylabel('Bending Moment [N*mm]')
ax[0].legend()

ax[1].plot(df['mm_Z'], df['N_Z'],'r-', label='AxialZ')
ax[1].set_xlabel('Displacement [mm]')
ax[1].set_ylabel('Axial Force [N]')
ax[1].legend()


#Get linear fit for the bending
fit_bendingX = np.polyfit(df['rad_X'], df['N*mm_X']/1000,1)
fit_bendingY = np.polyfit(df['rad_Y'], df['N*mm_Y']/1000,1)
fit_axialZ = np.polyfit(df['mm_Z']/1000, df['N_Z'],1)
print("Bending stiffness X [N*m/rad]:", fit_bendingX[0]) #0.015715502564662226
print("Bending stiffness Y [N*m/rad]:", fit_bendingY[0]) #0.015553204422264768
print("Axial stiffness Z [N/m]:", fit_axialZ[0]) #891.5517649350647


plt.show()