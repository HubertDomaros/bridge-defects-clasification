import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import PIL.Image as Image


def parse_training_log(log_text):
    """
    Parse training log and extract metrics for each epoch.

    Args:
        log_text (str): The training log text

    Returns:
        dict: Dictionary containing lists of metrics:
            - epoch: list of epoch numbers
            - train_loss: list of training losses
            - train_acc: list of training accuracies
            - val_loss: list of validation losses
            - val_acc: list of validation accuracies
            - best_model: list of booleans indicating if a new best model was saved
    """
    # Initialize results dictionary
    results = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_model': []
    }

    # Split log into lines
    lines = log_text.strip().split('\n')

    for i in range(0, len(lines)):  # Process 3 lines at a time
        if i + 2 >= len(lines):  # Check if we have enough lines left
            break

        # Extract epoch number
        # epoch_line = lines[i]
        # if not epoch_line.startswith('Epoch'):
        #     continue
        # epoch = int(epoch_line.split('/')[0].split()[-1])

        # Extract metrics
        metrics_line = lines[i + 1]
        train_metrics, val_metrics = metrics_line.split(' - ')

        # Parse training metrics
        train_loss = float(train_metrics.split('Loss: ')[1].split(',')[0])
        train_acc = float(train_metrics.split('Accuracy: ')[1].split('%')[0])

        # Parse validation metrics
        val_loss = float(val_metrics.split('Loss: ')[1].split(',')[0])
        val_acc = float(val_metrics.split('Accuracy: ')[1].split('%')[0])

        # Check if new best model was saved
        best_model = 'New best model saved!' in lines[i + 1]

        # Add to results
        #results['epoch'].append(epoch)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        results['best_model'].append(best_model)

    return results

with open(r'logs/training_progress_20250130_095709.log', 'r') as f:
    log = f.read()
results = parse_training_log(log)
df = pd.DataFrame(results)

sns.lineplot(x='epoch', y='train_loss', label='Train loss',  data=df)
sns.lineplot(x='epoch', y='val_loss', label='Val loss' , data=df)
plt.legend()
plt.grid()
plt.show()

sns.lineplot(x='epoch', y='train_acc', label='Train accuracy', data=df)
sns.lineplot(x='epoch', y='val_acc', label='Val accuracy', data=df)
plt.legend()
plt.grid()
plt.show()
print(df.tail())