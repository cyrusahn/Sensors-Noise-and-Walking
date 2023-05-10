import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, integrate

source = sys.argv[1]

walk = pd.read_csv(source)

# Calculate sample rate of dataset
walk['time_shifted'] = walk['time'].shift(periods=1, fill_value=0)
walk['sample_rate'] = 1 / (walk['time'] - walk['time_shifted'])
avg_sample_rate = walk['sample_rate'].mean()

# Find length of vector using acceleration of x, y, z components
walk['vector_length'] = np.sqrt(np.square(walk['ax']) + np.square(walk['ay']) + np.square(walk['az']))

# Find number of steps taken in unfiltered data
peaks, _ = signal.find_peaks(walk['vector_length'], distance=3)

# Plot unfiltered data
plt.figure(figsize=(15,10))
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title("Person Accelerometer Data")
plt.plot(walk['time'], walk['vector_length'], label='Unfiltered')

# Setup butterworth filter
b, a = signal.butter(1, .007, btype='lowpass', analog=False)
low_passed = signal.filtfilt(b, a, walk['vector_length'])

# Create new dataset of filtered data
filtered_data = {'time': walk['time'], 'vector_length': low_passed}
filtered = pd.DataFrame(filtered_data)

# Plot filtered data
plt.plot(walk['time'], low_passed, label='Filtered')
plt.legend()

# Find number of steps taken in filtered data
filtered_peaks, _ = signal.find_peaks(filtered['vector_length'], distance=3)
# filtered_peaks, _ = signal.find_peaks(filtered['vector_length'], width=.3)
# filtered_peaks, _ = signal.find_peaks(filtered['vector_length'], threshold=np.mean(low_passed))


num_steps_filtered = len(filtered_peaks)

total_walk_time = walk['time'].iloc[-1] - walk['time'].iloc[0] 
steps_per_sec = num_steps_filtered / total_walk_time

# Calculate change in velocity
filtered['change_vel_cummulative'] = integrate.cumtrapz(low_passed, filtered['time'], initial=0)
filtered['change_shift'] = filtered['change_vel_cummulative'].shift(periods=1, fill_value=0)
filtered['change_in_velocity'] = filtered['change_vel_cummulative'] - filtered['change_shift']
print(filtered)

print("average sample rate: ", avg_sample_rate, "hz")
print("steps per minute: ", steps_per_sec * 60)
# print("total speed: ", total_speed, "m/s")


plt.savefig('graphs.png')
# plt.show()
