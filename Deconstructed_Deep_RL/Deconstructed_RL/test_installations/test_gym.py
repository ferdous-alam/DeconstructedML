import gym
import glfw
import time

# env = gym.make('InvertedDoublePendulum-v2')
env = gym.make('Ant-v3')
env.reset()
for _ in range(1000):
    env.render()
    # time.sleep(0.01)
    env.step(env.action_space.sample())             # take a random action
env.close()
glfw.terminate()
