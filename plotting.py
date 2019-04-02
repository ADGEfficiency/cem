import matplotlib.pyplot as plt


def plot_history(history, env_id, num_episodes, expt_time):
    f, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 10))
    f.suptitle('{} {} samples {:0.0f} seconds'.format(env_id, num_episodes, expt_time))
    ax[0].plot(history['epoch'], history['avg_rew'], label='population')
    ax[0].plot(history['epoch'], history['avg_elites'], label='elite')
    ax[0].legend()
    ax[0].set_ylabel('average rewards')

    ax[1].plot(history['epoch'], history['std_rew'], label='population')
    ax[1].plot(history['epoch'], history['std_elites'], label='elites')
    ax[1].legend()
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('standard deivation rewards')

    f.savefig('./{}/learning.png'.format(env_id))
