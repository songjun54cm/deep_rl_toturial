# %%capture
# !apt install python-opengl
# !apt install ffmpeg
# !apt install xvfb
# !pip install pyvirtualdisplay
# !pip install pyglet==1.5.1
# !pip install -r https://raw.githubusercontent.com/huggingface/deep-rl-class/main/notebooks/unit4/requirements-unit4.txt


# Virtual display
from pyvirtualdisplay.display import Display

virtual_display = Display(visible=False, size=(1400, 900))
virtual_display.start()

# import numpy as np

# from collections import deque

# import matplotlib.pyplot as plt
# %matplotlib inline

# # PyTorch
import torch

# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.distributions import Categorical

# # Gym
# import gym
# import gym_pygame

# # Hugging Face Hub
# from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.
# import imageio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)