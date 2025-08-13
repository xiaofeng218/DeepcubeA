import numpy as np
import matplotlib.pyplot as plt

# Set up Chinese font support
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

K_values = list(range(30))
convergence_epochs = [0]*20 + [2, 1, 2, 3, 5, 8, 10, 13, 15, 16]

# Create figure
plt.figure(figsize=(12, 6))

# Plot line chart
plt.plot(K_values, convergence_epochs, 'o-', color='b', linewidth=2, markersize=8)

# Set chart title and axis labels
plt.title('Model Convergence Epochs vs K Values', fontsize=16)
plt.xlabel('K Value', fontsize=14)
plt.ylabel('Convergence Epochs', fontsize=14)

# Set ticks
plt.xticks([0] + list(range(1, 11)), fontsize=12)
plt.yticks(fontsize=12)

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.7)

# Save figure
save_path = 'k_convergence_epochs.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {save_path}")

# Display figure
plt.show()