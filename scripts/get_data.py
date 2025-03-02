"""
Setup train and test directories for images.
"""
import os
import zipfile
from typing import Tuple

from pathlib import Path
import requests

def download_data(web_link: str) -> Tuple[Path, Path]:
  """
  Download data from the internet as zip file and extract it
  to get train and test paths.

  Args:
    web_link: dataset url in the web (raw zip if using github links)

  Returns:
    A tuple of (train_path, test_path)

  Example usage:
    train_dir, test_dir = download_data(web_link=data_link)
  """

  # Setup path to data folder
  data_path = Path("data/")
  zip_name = web_link.split("/")[-1]
  data_name = zip_name.split(".")[0]

  zip_path = data_path / zip_name
  image_path = data_path / data_name

  # If the image folder doesn't exist, download it and prepare it...
  if image_path.is_dir():
      print(f"{image_path} directory exists.")
  else:
      print(f"Did not find {image_path} directory, creating one...")
      image_path.mkdir(parents=True, exist_ok=True)

  # Download web data
  with open(zip_path, "wb") as f:
      request = requests.get(web_link)
      print(f"Downloading {data_name} data...")
      f.write(request.content)

  # Unzip pizza, steak, sushi data
  with zipfile.ZipFile(zip_path, "r") as zip_ref:
      print(f"Unzipping {data_name} data...")
      zip_ref.extractall(image_path)

  # Remove zip file
  os.remove(zip_path)

  # Setup train and testing paths
  train_dir = image_path / "train"
  test_dir = image_path / "test"

  return train_dir, test_dir

def local_data(local_link: str) -> Tuple[Path, Path]:
  """
  Get train and test paths from a local dataset

  Args:
    local_link: dataset local url

  Returns:
    A tuple of (train_path, test_path)

  Example usage:
    train_dir, test_dir = local_data(local_link=data_link)
  """

  # Setup path to data folder
  image_path = Path(local_link)

  # If the image folder doesn't exist, raise an error
  assert image_path.is_dir(), f"Dataset doesn't exist at {local_link}, please enter a correct URL"


  # Setup train and testing paths
  train_dir = image_path / "train"
  test_dir = image_path / "test"

  return train_dir, test_dir
