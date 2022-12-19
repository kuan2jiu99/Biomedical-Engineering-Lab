import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import savgol_filter

t = []
y1 = []
y2 = []
with open("12-05_6fatigue.csv", 'r') as f:
    rows = csv.reader(f)
    for row in rows:
        t.append(int(row[0]))
        y1.append(int(row[2]))
        y2.append(int(row[3]))
        # print("")
y1 = np.array(y1) - np.mean(y1)
y2 = np.array(y2) - np.mean(y2)
window_size = 10
length = int((y1.shape[0] // window_size) * window_size)
t = t[:length]
y1 = y1[:length]
y2 = y2[:length]


yp = []
for i in range(y1.shape[0] // window_size):
    ymax = np.max(y1[i*window_size:i*window_size+window_size])
    ymin = np.min(y1[i*window_size:i*window_size+window_size])
    ypp = ymax - ymin
    yp = yp + [ypp]
    # yp = yp + [ypp for i in range(window_size)]
    # print("")
yp = np.array(yp)
th = 200
yp_sm = savgol_filter(yp, 51, 2, mode='nearest')
# yt = yp_sm - yp
action = (yp_sm > th).astype(int)

# plt.figure(figsize=(15,4), dpi=72)
# plt.plot(t[::window_size], yp, color='blue')
# plt.plot(t[::window_size], yp_sm, color='yellow')
# plt.plot(t[::window_size], action * th, color='green')
# plt.grid()
# plt.xlabel('t (s)', fontsize=14)
# plt.ylabel('amplitude', fontsize=14)
# plt.show()

start = ((action[1:] - action[:-1]) == 1).astype(int)
end = ((action[1:] - action[:-1]) == -1).astype(int)

fs = 480.0  # Sample frequency (Hz)
f0 = 60.0  # Frequency to be removed from signal (Hz)
Q = 30.0  # Quality factor
# Design notch filter
b, a = signal.iirnotch(f0, Q, fs)

# y1_denoised = signal.filtfilt(b, a, y1)
# y1 = y1_denoised

f, t, Zxx = signal.stft(y1, fs, nperseg=1000)
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.max(np.abs(Zxx)), shading='gouraud')
# plt.pcolormesh(t, f[:10], np.abs(Zxx)[:10], vmin=0, vmax=np.max(np.abs(Zxx)), shading='gouraud')
# energy_cumsum = np.cumsum(np.abs(Zxx), axis=0)
# Zxx_max = np.abs(Zxx)[np.where(energy_cumsum>np.max(energy_cumsum)/2)[0][0]]   
Zxx_max = f[np.argmax(np.abs(Zxx), axis=0)]
Zxx_max_edited = []
Zxx_prev = -1
for i in range(len(Zxx_max)):
    if Zxx_max[i] > 50:
        Zxx_max_edited.append(Zxx_prev)
    else:
        Zxx_max_edited.append(Zxx_max[i])
        Zxx_prev = Zxx_max[i]
weighted_prev = 0
Zxx_max_ave = []
for i in range(len(Zxx_max_edited)):
    weighted_prev = weighted_prev * 0.97 + Zxx_max_edited[i] * 0.03
    Zxx_max_ave.append(weighted_prev)

# window_size2 = 150
# Zxx_max_final = [0 for i in range(window_size2)]
# slide = Zxx_max_ave[:window_size2]
# for i in range(window_size2, len(Zxx_max_ave)):
#     slide.append(Zxx_max_ave[i])
#     slide = slide[1:]
#     Zxx_max_final.append(np.mean(slide))

plt.plot(t, Zxx_max_edited)

plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()