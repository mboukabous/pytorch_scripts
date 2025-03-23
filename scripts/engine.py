"""
Contains functions for training and testing a PyTorch model.
"""

from typing import Tuple, Dict, List, Any
from tqdm.auto import tqdm
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Args:
      model: A PyTorch model to be trained.
      dataloader: A DataLoader instance for training data.
      loss_fn: Loss function.
      optimizer: Optimizer.
      device: Target device (e.g. "cuda" or "cpu").

    Returns:
      A tuple of (train_loss, train_accuracy).
    """
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Zero gradients, backward pass, optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy for this batch
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float, float, float, float, np.ndarray]:
    """Tests a PyTorch model for a single epoch and computes metrics.

    In addition to loss and accuracy, this function computes
    macro-averaged precision, recall, F1-score, and the confusion matrix.

    Args:
      model: A PyTorch model to be tested.
      dataloader: A DataLoader instance for testing data.
      loss_fn: Loss function.
      device: Target device (e.g. "cuda" or "cpu").

    Returns:
      A tuple containing:
        - test_loss: Average test loss.
        - test_acc: Average test accuracy.
        - macro_precision: Macro-averaged precision.
        - macro_recall: Macro-averaged recall.
        - macro_f1: Macro-averaged F1-score.
        - cm: Confusion matrix (NumPy array).
    """
    model.eval()
    test_loss, test_acc = 0, 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Get predicted class labels
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

            # Accumulate predictions and true labels for metric computation
            all_preds.extend(test_pred_labels.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    # Convert accumulated lists to NumPy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute weighted-averaged metrics using scikit-learn
    weighted_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    weighted_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return test_loss, test_acc, weighted_precision, weighted_recall, weighted_f1, cm

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List[Any]]:
    """Trains and tests a PyTorch model over multiple epochs.

    Computes training and testing loss, accuracy, macro-averaged
    precision, recall, F1-score, and confusion matrix at each epoch.

    Args:
      model: A PyTorch model.
      train_dataloader: DataLoader for training.
      test_dataloader: DataLoader for testing.
      optimizer: Optimizer.
      loss_fn: Loss function.
      epochs: Number of training epochs.
      device: Target device (e.g. "cuda" or "cpu").

    Returns:
      A dictionary of metrics collected over epochs.
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "weighted_precision": [],
        "weighted_recall": [],
        "weighted_f1": [],
        "confusion_matrix": []
    }

    for epoch in tqdm(range(1, epochs + 1), desc="Training Epochs"):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc, weighted_precision, weighted_recall, weighted_f1, cm = test_step(model=model,
                                                                                     dataloader=test_dataloader,
                                                                                     loss_fn=loss_fn,
                                                                                     device=device)

        tqdm.write(f"Epoch {epoch}/{epochs} | "
                   f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
                   f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f} | "
                   f"precision: {weighted_precision:.4f} | recall: {weighted_recall:.4f} | f1-score: {weighted_f1:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["weighted_precision"].append(weighted_precision)
        results["weighted_recall"].append(weighted_recall)
        results["weighted_f1"].append(weighted_f1)
        results["confusion_matrix"].append(cm)

    return results
