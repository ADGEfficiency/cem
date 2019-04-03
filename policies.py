import numpy as np


def setup_policy(env, theta):
    if env.action_space.type == 'discrete':
        return DiscretePolicy(env, theta)

    elif env.action_space.type == 'continuous':
        return ContinuousPolicy(env, theta)

    else:
        raise ValueError


class ContinuousPolicy():
    def __init__(
            self, env, theta
    ):
        self.env = env

        obs_shape = env.observation_space.shape[0]
        act_shape = env.action_space.shape[0]
        assert(len(theta) == (obs_shape + 1) * act_shape)

        self.parameter_dim = obs_shape * act_shape
        self.b = theta[self.parameter_dim:]
        self.W = theta[:self.parameter_dim].reshape(obs_shape, act_shape)

    def act(self, observation):
        return np.clip(
            observation.dot(self.W) + self.b,
            self.env.action_space.low,
            self.env.action_space.high
        )


class DiscretePolicy():
    def __init__(
            self, env, theta
    ):
        self.env = env

        obs_shape = env.observation_space.shape[0]
        num_actions = env.action_space.n
        assert(len(theta) == (obs_shape + 1) * num_actions)

        self.parameter_dim = obs_shape * num_actions
        self.W = theta[:self.parameter_dim].reshape(obs_shape, num_actions)
        self.b = theta[self.parameter_dim:]

    def act(self, observation):
        y = observation.dot(self.W) + self.b
        action = np.argmax(y)
        return action
