import gym
import minitouch.env

def make_dmc(env_domain):
    env = gym.make(env_domain)
    return env
