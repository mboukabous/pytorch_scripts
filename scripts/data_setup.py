"""
Contains functionality for creating PyTorch DataLoader's for
Image classification data.
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str, test_dir: str, batch_size: int,
                       train_transform: transforms.Compose=None, test_transform: transforms.Compose=None,
                       img_size: tuple=None, num_workers: int=NUM_WORKERS):
  """
  Creates training and testing DataLoders
  Takes in training and testing dir paths with other params
  And turns them into PyTorch Dataset and DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    batch_size: Number of samples per batch in each DataLoader.
    train_transform: transforms to perform on training data.
    test_transform: transforms to perform on testing data.
    img_size: a tuple used to create a simple transform in case train_transform or test_transform is None.
    num_workers: An integer for number of workers per DataLoader (default is CPU number available).

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).

  Example usage:
    train_dataloader, test_dataloader, class_names = create_dataloaders(
      train_dir=path/to/train_dir, test_dir=path/to/test_dir,
      train_transform=some_transforms_compose, test_transform=some_transforms_compose,
      batch_size=32, num_workers=4)
  """
  # Check is a transform exist, otherwise create a simple one using img_size
  if not train_transform:
    train_transform = transforms.Compose([
      transforms.Resize(img_size),
      transforms.ToTensor()
    ])
  if not test_transform:
    test_transform = transforms.Compose([
      transforms.Resize(img_size),
      transforms.ToTensor()
    ])

  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                    transform=train_transform, # transforms to perform on data (images)
                                    target_transform=None) # transforms to perform on labels (if necessary)
  test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)

  # Get the class_names
  class_names = train_data.classes

  # Turn images into DataLoaders
  train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch_size, # how many samples per batch?
                                num_workers=num_workers, # how many subprocesses to use for data loading?
                                shuffle=True, # shuffle the data?
                                pin_memory=True) # enable fast data transfer to CUDA-GPU
  test_dataloader = DataLoader(dataset=test_data,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               shuffle=False, # don't usually need to shuffle testing data
                               pin_memory=True)

  return train_dataloader, test_dataloader, class_names
