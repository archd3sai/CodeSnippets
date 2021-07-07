# libraries
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120 #resolution
matplotlib.rcParams['figure.figsize'] = (8,6) #figure size

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns

# for a single plot
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(df['accuracy'], linewidth = 1, label = "Accuracy", marker = 'o', markersize = 7)
ax.plot(df['val_accuracy'], linewidth = 1, label = "Validation Accuracy")

# for multiple subplots
fig, ax = plt.subplots(1,2, figsize=(5, 3))

# for multiple plots and without middle xticks
fig, ax = plt.subplots(1,2, figsize=(5, 3), sharex=True, sharey=True)

ax[0].plot(autoencoder_w1_history['accuracy'], linewidth = 1, label = "Accuracy")
ax[1].plot(autoencoder_w1_history['val_accuracy'], linewidth = 1, label = "Validation Accuracy")

fig, ax = plt.subplots(2,2, figsize=(5, 3))
ax[0,0].plot(df['accuracy'], linewidth = 1, label = "Accuracy")
ax[1,1].plot(df['val_accuracy'], linewidth = 1, label = "Validation Accuracy")

# setting legend and its font size
ax.legend(loc = 'best', prop={'size': 8}, fontize=10)

# setting title
ax.set_title('Model Accuracy', size = 12)

# setting labels
ax.set_xlabel('Epochs', size = 10)
ax.set_ylabel('Accuracy', size = 10);

# setting label size
ax.tick_params(axis = 'both', labelsize = 8)

# for limits
ax.set_xlim(2,10)
ax.set_ylim(2,10)

# for rotation 
ax.xaxis.set_tick_params(rotation=0, labelsize = 8)

# for seaborn plots specify axis as ax
sns.lineplot(range(10), df['accuracy'], ax = ax, linewidth = 1, label = "Accuracy")
sns.lineplot(range(10), df['val_accuracy'], ax = ax, linewidth = 1, label = "Validation Accuracy")

# for common x and y label in subplots
fig.text(0.5, 0, 'common X', ha='center')
fig.text(0, 0.5, 'common Y', va='center', rotation='vertical')

# for common title
plt.suptitle('common title)

# for better layout
plt.tight_layout()

# to save image
fig.savefig('fig.png', dpi=fig.dpi)

# to add text next to points in scatter plot
for i in range(df.shape[0]):
    ax.text(df.loc[i, 'x_col']+1, df.loc[i, 'y_col'], df.loc[i, 'unique_id'],
           size = 'small', color = 'black', weight = 'semibold', horizontalalignment='left')     

