import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

def get_smoothed_data(df, column, weight=0.9):
    smoothed = []
    last = df[column].iloc[0]
    for value in df[column]:
        smoothed_val = last * weight + (1 - weight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def extract_tensorboard_data(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    tags = ea.Tags()['scalars']
    data = {}
    for tag in tags:
        df = pd.DataFrame(ea.Scalars(tag)).drop(columns=['wall_time'])
        df.columns = ['step', 'value']
        data[tag] = df
    return data

def plot_training_results(log_dir, smoothing=0.9, save_name="plot.webp"):
    data_dict = extract_tensorboard_data(log_dir)
    
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 20,
        "axes.titlesize": 24,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 20))

    val_max_step = data_dict['Val Loss']['step'].max()
    
    def scale_to_epochs(df):
        df = df.copy()
        max_s = df['step'].max()
        df['epoch_step'] = (df['step'] / max_s) * val_max_step
        return df

    if 'Loss' in data_dict:
        df_loss = scale_to_epochs(data_dict['Loss'])
        df_loss['Smooth'] = get_smoothed_data(df_loss, 'value', smoothing)
        sns.lineplot(data=df_loss, x='epoch_step', y='value', ax=ax1, alpha=0.15, color='#1f77b4')
        sns.lineplot(data=df_loss, x='epoch_step', y='Smooth', ax=ax1, label='Train Loss', color='#1f77b4', linewidth=3)

    if 'Val Loss' in data_dict:
        sns.lineplot(data=data_dict['Val Loss'], x='step', y='value', ax=ax1, label='Val Loss', 
                     color='#d62728', marker='o', markersize=10, linewidth=3)

    ax1.set_title("Training & Validation Loss", fontweight='bold', pad=25)
    ax1.set_xlabel("Epoch", labelpad=15)
    ax1.set_ylabel("Loss Value", labelpad=15)
    ax1.legend(loc='upper right')

    if 'Accuracy' in data_dict:
        df_acc = scale_to_epochs(data_dict['Accuracy'])
        smooth_acc = get_smoothed_data(df_acc, 'value', smoothing)
        sns.lineplot(x=df_acc['epoch_step'], y=smooth_acc, ax=ax2, label='Train Acc', color='#2ca02c', alpha=0.4, linewidth=2)

    if 'Val Accuracy' in data_dict:
        sns.lineplot(data=data_dict['Val Accuracy'], x='step', y='value', ax=ax2, label='Val Acc', 
                     color='#2ca02c', marker='s', markersize=8, linewidth=3)

    if 'Val F1' in data_dict:
        sns.lineplot(data=data_dict['Val F1'], x='step', y='value', ax=ax2, label='Val F1', 
                     color='#9467bd', marker='D', markersize=8, linewidth=3)

    ax2.set_title("Model Performance Metrics", fontweight='bold', pad=25)
    ax2.set_xlabel("Epoch", labelpad=15)
    ax2.set_ylabel("Score (0.0 - 1.0)", labelpad=15)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc='lower right')

    plt.tight_layout(pad=8.0)
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Professional plot saved as: {save_name}")
    plt.show()

if __name__ == "__main__":
    LOG_PATH = "logsOld/PathClassifierEdgeNext/20240529-092849" 
    plot_training_results(LOG_PATH)