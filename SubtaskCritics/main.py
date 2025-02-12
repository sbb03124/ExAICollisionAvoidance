# ライブラリのimport
import shutil
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
tf.compat.v1.experimental.output_all_intermediates(True)
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config = config))

# ファイルの読み込み
import train
import environment
import subtaskcritics

# main：学習用, test：テスト用
def main(visualize, reward_params):
    nb_trainstep = 5000000
    window_length = 5
    batch_size = 1024
    nb_wormup_actor  = batch_size*100 # max([int(nb_trainstep*0.025), batch_size*2])
    nb_wormup_critic = batch_size*20 # max([int(nb_trainstep*0.005), batch_size*2])

    learning_rate       = 0.0010
    target_model_update = 0.0010
    train_interval = 100
    visualize = visualize
    weight_save_interval = int(nb_trainstep/100)
    smoothing_wormup = int(nb_trainstep*0.8)
    smoothing_gain = 0

    log_file = 'params_{}_{}_{}_{}_{}_log'.format(
        *reward_params
    ).replace('.','-') + '.csv'

    ######################################################
    env = environment.environment(reward_params)
    obs_n = (window_length,) + env.observation_space.shape
    act_n = env.action_space.shape


    agent = subtaskcritics.Agent(
        wp_shape = (5,3),
        oth_shape = (5*10,),
        act_shape = (1,),
        random = subtaskcritics.OrnsteinUhlenbeckProcess(
            size=act_n,
            theta=0.01,
            mu=0.0,
            dt=1, # 15.0から変更
            sigma=0.30, # 0.06から変更
            sigma_min=0.01,
            n_steps_annealing=int(nb_trainstep*0.8),
            nb_wormup = int(nb_trainstep*0.2)
        ),
        memory = subtaskcritics.Memory(maxlen=1000000, window_length=window_length),
        actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=0.5*1e-8),
        critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=0.5*1e-8),
        batch_size = batch_size,
        gamma = 0.98,
        target_model_update = target_model_update,
    )

    train.train(agent, env, nb_trainstep, train_interval, nb_wormup_actor, nb_wormup_critic, weight_save_interval, log_file, visualize, smoothing_wormup, smoothing_gain )

    return agent

def debug():
    nb_trainstep = 1000
    window_length = 5
    batch_size = 10
    nb_wormup_actor  = 100
    nb_wormup_critic =  50

    learning_rate       = 0.0010
    target_model_update = 0.0050
    train_interval = 10
    visualize = False
    weight_save_interval = 1000
    smoothing_wormup = int(nb_trainstep*0.8)
    smoothing_gain = 1e-3

    log_file = 'log.csv'

    ######################################################
    env = environment.environment()
    obs_n = (window_length,) + env.observation_space.shape
    act_n = env.action_space.shape


    agent = subtaskcritics.Agent(
        wp_shape = (5,3),
        oth_shape = (5*10,),
        act_shape = (1,),
        random = subtaskcritics.OrnsteinUhlenbeckProcess(
            size=act_n,
            theta=0.01,
            mu=0.0,
            dt=30.0, # 15.0から変更
            sigma=0.12, # 0.06から変更
            sigma_min=0.005,
            n_steps_annealing=int(nb_trainstep*0.8),
            nb_wormup = nb_wormup_actor
        ),
        memory = subtaskcritics.Memory(maxlen=batch_size*100, window_length=window_length),
        actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=0.5*1e-8),
        critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=0.5*1e-8),
        batch_size = batch_size,
        gamma = 0.99,
        target_model_update = target_model_update,
    )

    train.train(agent, env, nb_trainstep, train_interval, nb_wormup_actor, nb_wormup_critic, weight_save_interval, log_file, visualize, smoothing_wormup, smoothing_gain )

    return agent

def test(weight_file, nb_test, visualize, save=False, params=None):
    window_length = 5
    learning_rate = 1e-3
    batch_size = 100
    target_model_update = 1e-3
    ##############################################################
    ##############################################################
    env = environment.environment(params)
    env.test_mode = True
    if save:
        from wrapper import RecordVideo
        from datetime import datetime
        import os
        _dir = datetime.now().strftime('video%Y%m%d%H%M%S/')
        os.makedirs(_dir)
        env = RecordVideo( env, _dir, name_prefix='evalu_each' )

    obs_n = (window_length,) + env.observation_space.shape
    act_n = env.action_space.shape
    agent = subtaskcritics.Agent(
        wp_shape = (5,3),
        oth_shape = (5*10,),
        act_shape = (1,),
        random = subtaskcritics.OrnsteinUhlenbeckProcess(
            size=act_n,
            theta=0.01,
            mu=0.0,
            dt=30.0, # 15.0から変更
            sigma=0.12, # 0.06から変更
            sigma_min=0.005,
            n_steps_annealing = 1,
            nb_wormup = 0
        ),
        memory = subtaskcritics.Memory(maxlen=window_length, window_length=window_length),
        actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=0.5*1e-8),
        critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=0.5*1e-8),
        batch_size = batch_size,
        gamma = 0.99,
        target_model_update = target_model_update,
        training=False
    )
    if not weight_file=='dummy':
        agent.load_weights(weight_file)
    
    print('start test')
    train.test(agent, env, nb_test, visualize=visualize)
    # if input(f'move weight file to [test{env.exe_time}]\n[Y/n]> ').upper()=='Y':
    if True:
        shutil.move(
            weight_file.replace('.h5','_actor.h5'),
            f'test{env.exe_time}/'+weight_file.replace('.h5','_actor.h5'),
        )
        shutil.move(
            weight_file.replace('.h5','_critic.h5'),
            f'test{env.exe_time}/'+weight_file.replace('.h5','_critic.h5'),
        )

import argparse
def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--local', action='store_true')
    
    parser.add_argument('--epi', default=25)
    parser.add_argument('-t', '--test', default=None)
    parser.add_argument('--init', default=None)
    parser.add_argument('--exclude', default=None)
    parser.add_argument('--agent', default='MADDPG')
    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()

if __name__=='__main__':
    _params  = [ 0.220, 0.85, 0.15, 0.030, 45, ]
    # _trained = main(False, _params)
    # fname = 'trained_{}_{}_{}_{}_{}'.format(
    #         *_params
    #     ).replace('.','-') + '.h5'
    # _trained.save_weights(fname)
    test('trained_0-22_0-85_0-15_0-03_45.h5', 200, False, save=False, params=_params)