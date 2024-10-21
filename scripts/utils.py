"""
Contains various utility functions for PyTorch model training.
"""

import torch
from pathlib import Path
import pandas as pd

def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
  """
  Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(), f=model_save_path)

def save_results(results: dict, target_dir: str, model_name: str):
  """Saves a PyTorch model results to a target directory.

  Args:
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model results.

  Example usage:
    save_model(model=model_name,
               target_dir="results",
               model_name="model_name_results")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  # Create model results save path
  model_save_path = target_dir_path / model_name

  # Creating a dataframe from results dictionnary
  results_df = pd.DataFrame(results)

  # Save the model state_dict()
  print(f"[INFO] Saving results to: {model_save_path}")
  results_df.to_csv(model_save_path, index=False)

def load_model(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
  """
  Load a PyTorch model parameters from a target path.

  Args:
    model_path: A path for loading the model from.

  Example usage:
    load_model(model_path="models/model_name.pth")
  """
  # Load the model state_dict()
  print(f"[INFO] Loading model from: {model_path}")
  model.load_state_dict(torch.load(model_path, weights_only=False))
  return model
