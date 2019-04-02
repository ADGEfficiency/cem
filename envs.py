import gym


def setup_env(env_id):
    if env_id == 'cartpole':
        return setup_cartpole()

    elif env_id == 'pendulum':
        return setup_pendulum()

    else:
        raise ValueError('env {} not supported'.format(env_id))


def setup_pendulum():
    env = gym.make('Pendulum-v0')
    obs_shape = env.observation_space.shape[0]
    act_shape = env.action_space.shape[0]

    env.action_space.type = 'continuous'
    return env, obs_shape, act_shape


def setup_cartpole():
    env = gym.make('CartPole-v0')
    obs_shape = env.observation_space.shape[0]
    act_shape = env.action_space.n

    env.action_space.type = 'discrete'

    #  shape is empty tuple in the gym env
    env.action_space.shape = (1,)
    return env, obs_shape, act_shape
