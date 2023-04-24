# Test numpy
print("Importing NumPy")
try:
    import numpy as np
    print("NumPy loaded successfully.")
except:
    print("Could not import numpy successfully.")

# Test pytorch
print("Import PyTorch")
try:
    import torch
    print("PyTorch imported successfully.")
    print(f"Is cuda available? {torch.cuda.is_available()}")
except:
    print("Could not import torch")

# Test wandb
print("Import wandb")
try:
    import wandb
    print("WandB imported successfully.")
except:
    print("Could not import wandb")

# Test learned_cost_maps
print("Importing learned_cost_maps")
try:
    from learned_cost_maps.trainer.utils import *
    print("Learned cost maps imported successfully")
except:
    print("Could not import learned_cost_maps")