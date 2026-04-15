import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create the output directory if it doesn't exist
output_dir = r"c:\Users\Acer\OneDrive\Desktop\NeuroGAN\app\frontend\public\assets\charts"
os.makedirs(output_dir, exist_ok=True)

# Styling for premium look
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial']
primary_color = '#60A5FA'  # Blue-400
accent_color = '#F472B6'  # Pink-400
success_color = '#34D399' # Emerald-400

# 1. Class Distribution Chart
classes = ['Class 0', 'Class 1\n(Minority)', 'Class 2', 'Class 3']
counts = [724, 49, 2566, 1781]

plt.figure(figsize=(10, 6))
bars = plt.bar(classes, counts, color=[primary_color, accent_color, primary_color, primary_color])
plt.title('Original Dataset Class Distribution', fontsize=16, pad=20, color='white', fontweight='bold')
plt.ylabel('Number of Samples', fontsize=12, color='#9CA3AF')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 50, int(yval), ha='center', va='bottom', color='white', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, transparent=True)
plt.close()

# 2. F1-Score Performance Chart
f1_scores = [0.91, 1.00, 0.96, 0.94]

plt.figure(figsize=(10, 6))
bars = plt.bar(classes, f1_scores, color=success_color)
plt.title('CNN Model Performance (F1-Score by Class)', fontsize=16, pad=20, color='white', fontweight='bold')
plt.ylabel('F1-Score', fontsize=12, color='#9CA3AF')
plt.ylim(0.8, 1.05)  # Zoom in on relevant range
plt.axhline(y=0.9528, color=accent_color, linestyle='--', linewidth=2, label='Overall Weighted F1 (95.28%)')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend()

# Add value labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.2f}", ha='center', va='bottom', color='white', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'f1_scores.png'), dpi=300, transparent=True)
plt.close()

print("Charts generated successfully at:", output_dir)
