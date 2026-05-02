from pyvirtualdisplay.display import Display

virtual_display = Display(visible=False, size=(1400, 900))
virtual_display.start()

from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("LunarLander-v2", n_envs=16)