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
        dict: Dictionary containing lists of metrics
    """
    results = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_model': []
    }
    
    lines = log_text.strip().split('\n')
    for i in range(0, len(lines)):
        if i + 2 >= len(lines):
            break
        epoch_line = lines[i]
        if not epoch_line.startswith('Epoch'):
            continue
            
        # Fixed epoch parsing
        epoch = int(epoch_line.split('/')[0].split()[-1])
        
        metrics_line = lines[i + 1]
        train_metrics, val_metrics = metrics_line.split(' - ')
        
        train_loss = float(train_metrics.split('Loss: ')[1].split(',')[0])
        train_acc = float(train_metrics.split('Accuracy: ')[1].split('%')[0])
        
        val_loss = float(val_metrics.split('Loss: ')[1].split(',')[0])
        val_acc = float(val_metrics.split('Accuracy: ')[1].split('%')[0])
        
        best_model = 'New best model saved!' in lines[i + 1]
        
        # Added missing epoch append
        results['epoch'].append(epoch)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        results['best_model'].append(best_model)
    
    return results

with open('logs/training_progress_20250130_181620.log', 'r') as f:
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