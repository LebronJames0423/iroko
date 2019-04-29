import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import itertools
import os
import json
import numpy as np
import seaborn as sns


def running_mean(x, N=100):
    if (len(x) < N or N < 1):
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[int(N):] - cumsum[:-int(N)]) / float(N)

reward_max = -np.inf
reward_min = np.inf
arr = []
with open('./workspace/fl/reward.txt', 'r') as f:
    for line in f:
        s = line.split('\n')
        num = float(s[0])
        arr.append(num)

if np.amax(arr) > reward_max:
    reward_max = np.amax(arr)
if np.amin(arr) < reward_min:
    reward_min = np.amin(arr)

mean_smoothing = 40000 / 100
res = running_mean((arr - reward_min) / (reward_max - reward_min), 10)

plt.plot(res)
plt.show()
