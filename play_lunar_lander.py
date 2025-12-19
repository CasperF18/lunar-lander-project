import gymnasium as gym
import time

def main():
    env = gym.make("LunarLander-v3", render_mode="human")
    obs, info = env.reset(seed=0)

    done = False
    while not done:
        action = env.action_space.sample()  # random actions
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        time.sleep(1 / 60)  # slow enough to see

    env.close()

if __name__ == "__main__":
    main()
