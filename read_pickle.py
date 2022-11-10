import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

rewards = pd.read_pickle(r'evaluation_rewards.pkl')
steps = pd.read_pickle(r'evaluation_steps.pkl')
# rewards = savgol_filter(rewards, 51, 3) # window size 51, polynomial order 3
plt.plot(steps, rewards)
plt.show()
