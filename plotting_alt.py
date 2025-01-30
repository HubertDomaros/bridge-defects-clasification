import matplotlib.pyplot as plt
import re

# Parse the log data
train_loss = []
train_acc = []
val_loss = []
val_acc = []

# Regular expression pattern to extract metrics
pattern = r'Train Loss: ([\d.]+), Train Accuracy: ([\d.]+)% - Val Loss: ([\d.]+), Val Accuracy: ([\d.]+)%'

# Read the log line by line
with open('logs/training_progress_20250130_181620.log', 'r') as f:
    log_lines = f.read()
    print(log_lines)

for line in log_lines:
    match = re.match(pattern, line)
    if match:
        train_loss.append(float(match.group(1)))
        train_acc.append(float(match.group(2)))
        val_loss.append(float(match.group(3)))
        val_acc.append(float(match.group(4)))

# Create epochs array
epochs = range(1, len(train_loss) + 1)

# Create the plot with two subplots
plt.figure(figsize=(12, 5))

# Loss subplot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy subplot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Adjust layout and display
plt.tight_layout()
plt.show()