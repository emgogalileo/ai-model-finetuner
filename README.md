# AI Model Fine-Tuner

A practical machine learning fine-tuning toolkit written in **Python** with scikit-learn.  
Demonstrates a complete ML lifecycle: load → preprocess → search → train → evaluate → checkpoint.

## Tech Stack
- **Language**: Python 3.11+
- **ML**: scikit-learn 1.4+, NumPy 1.26+
- **Models**: Logistic Regression, Random Forest, SVM
- **Evaluation**: Accuracy, ROC-AUC, Classification Report, Confusion Matrix

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Quick demo (logistic regression, 10 epochs)
python trainer.py --demo

# Random Forest with hyperparameter search
python trainer.py --model random_forest --epochs 20 --search

# All options
python trainer.py --model svm --epochs 15 --output checkpoints/
```

## CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `logistic` | Model type: `logistic`, `random_forest`, `svm` |
| `--epochs` | `10` | Number of training loop iterations |
| `--search` | off | Enable GridSearchCV hyperparameter search |
| `--output` | `checkpoints/` | Directory for model checkpoints |
| `--demo` | off | Run with all defaults |

## Training Output

```
08:00:00 [INFO] 🤖  AI Fine-Tuner — model=logistic  epochs=10
08:00:00 [INFO] 📂  Loading dataset…
08:00:00 [INFO] 🏋️  Training…
08:00:00 [INFO]   Epoch  1/10 — loss: 0.6124  train_acc: 0.8750  val_acc: 0.8824
...
08:00:01 [INFO] 📊  Evaluating on test set…
08:00:01 [INFO]   Test accuracy: 0.9825  ROC-AUC: 0.9961
08:00:01 [INFO]   💾  Model saved → checkpoints/logistic_1715535600.pkl
08:00:01 [INFO] ✅  Fine-tuning complete.
```

## Checkpoints

Each training run saves two files to the output directory:
- `{model}_{timestamp}.pkl` — serialized sklearn model
- `{model}_{timestamp}_metadata.json` — full training history, params, and eval metrics

## Project Structure

```
trainer.py          # Main training script + CLI
requirements.txt    # Python dependencies
checkpoints/        # Saved model checkpoints (git-ignored)
```

## Author
Emmanuel García — [emmanuelg@allcognition.com](mailto:emmanuelg@allcognition.com)
