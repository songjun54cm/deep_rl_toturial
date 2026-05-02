# !apt install swig cmake
# !pip install stable-baselines3 swig gymnasium[box2d] huggingface_sb3
# !apt-get update
# !apt install python3-opengl
# !apt install ffmpeg
# !apt install xvfb
# !pip3 install pyvirtualdisplay
# !pip install shimmy


import gymnasium as gym
from huggingface_hub import notebook_login
from huggingface_sb3 import package_to_hub
from pyvirtualdisplay.display import Display
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

virtual_display = Display(visible=False, size=(1400, 900))
virtual_display.start()

env_name = "LunarLander-v3"
env = make_vec_env(env_name, n_envs=16, seed=0, vec_env_cls=SubprocVecEnv)

model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
)

# Train it for 2,000,000 timesteps
model.learn(total_timesteps=2000000, progress_bar=True)
# Save the model
model_name = "ppo-LunarLander-v3"
model.save(model_name)

eval_env = Monitor(gym.make("LunarLander-v3", render_mode="rgb_array"))
mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=10, deterministic=True
)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

notebook_login()
# !git config --global credential.helper store


env_id = "LunarLander-v3"
model_architecture = "PPO"
repo_id = "junsong9001/ppo-LunarLander-v3"
commit_message = "Upload PPO LunarLander-v3 trained agent"

eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id, render_mode="rgb_array"))])

package_to_hub(
    model=model,  # Our trained model
    model_name=model_name,  # The name of our trained model
    model_architecture=model_architecture,  # The model architecture we used: in our case PPO
    env_id=env_id,  # Name of the environment
    eval_env=eval_env,  # Evaluation Environment
    repo_id=repo_id,  # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
    commit_message=commit_message,
)