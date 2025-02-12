import os
import csv
import time
import numpy as np
from copy import deepcopy
from datetime import datetime

def train(agent, env, nb_trainstep, train_interval, nb_wormup_actor, nb_wormup_critic, weight_save_interval, log_file = None, visualize=False, smoothing_wormup=None, smoothing_gain=0):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    save_dir = 'log'+timestamp
    if log_file is not None:
        log_file = timestamp + log_file
    os.makedirs(save_dir, exist_ok=True)
    train_start = datetime.now()

    done = True
    episode=0
    step = 0
    log_reward = [ ]
    log_states0  = [ ]
    log_states1  = [ ]
    log_done  = [ ]
    time_flagment = time.perf_counter()

    losses = [None,None]
    if log_file is not None:
        log_file = save_dir + '/' +log_file
        with open(log_file, 'w', newline='') as f:
            cols = ['episode', 'steps', 'reward', 'actor_loss', 'critic_loss']
            for name in agent.actor.trainable_weights:
                cols += [
                    name.name+'_min', name.name+'_max', name.name+'_mean'
                ]
            for name in agent.critic.trainable_weights:
                cols += [
                    name.name+'_min', name.name+'_max', name.name+'_mean'
                ]
            writer = csv.writer(f)
            writer.writerow(cols)
    try:
        while step < nb_trainstep:
            step += 1
            # print(step)
            if done:
                episode += 1
                episode_step = 0
                done = False
                state0 = env.reset()
                log_action = [ ]
                log_reward = [ ]
                log_states0  = [ ]
                log_states1  = [ ]
                log_done  = [ ]
                if visualize: env.render()
                time_flagment = time.perf_counter()
            episode_step += 1
            action = agent.get_action(state0)

            state1, reward, done, _ = env.step(action)
            if visualize: env.render()

            agent.append( state0, state1, reward, done, action )
            log_action.append(action)
            log_reward.append(reward)
            log_done.append(done)
            log_states0.append(state0)
            log_states1.append(state1)
            
            state0 = deepcopy(state1)
            
            if done:
                print(f'episode : {episode} [ {step}/{nb_trainstep} ({step/nb_trainstep*100:06.2f}%) ]')
                print('step per second > {:.1f}'.format(episode_step/(time.perf_counter()-time_flagment)))
                log_data = [episode]

                text = f''
                text += f'[steps:{len(log_done)}]'
                text += f'[reward {sum(log_reward):7.2f}]'
                text += f'[action : {np.array(log_action).min():3.1f}~{np.array(log_action).max():3.1f}]'
                text += f'[state : {np.array(log_states0).min():.2f}~{np.array(log_states0).max():.2f}]'
                text += f'[reward : {np.array(log_reward).min():.2f}~{np.array(log_reward).max():.2f}]'
                print(text)
                
                log_data += [
                    len(log_done), sum(log_reward), losses[0], losses[1]
                ]
                log_data += agent.grad_info[1]
                print('\n')
                if log_file is not None:
                    with open(log_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(log_data)


            if step%train_interval == 0 and (step>=nb_wormup_actor or step>=nb_wormup_critic):
                print('[update start]')
                actor_loss, critic_loss = agent.train(
                    train_actor=(step>=nb_wormup_actor)and(step%(train_interval*2)==0), trian_critic=step>=nb_wormup_critic,
                    smoothing_gain = smoothing_gain if (smoothing_wormup is not None)and(step>smoothing_wormup) else 0
                )
                print(
                    'actor_loss {} ({}), critic_loss {} ({})'.format(
                        f'{actor_loss.numpy():.5f}' if actor_loss is not None else 'None',
                        'training' if step>=nb_wormup_actor else 'warming',
                        f'{critic_loss.numpy():.5f}' if critic_loss is not None else 'None',
                        'training' if step>=nb_wormup_critic else 'warming',
                    )
                )
                losses[0] = actor_loss.numpy() if (actor_loss is not None and step>=nb_wormup_actor) else None
                losses[1] = critic_loss.numpy() if (critic_loss is not None and step>=nb_wormup_critic) else None

                print('[update end]')

            if step%weight_save_interval == 0:
                fname = datetime.now().strftime(save_dir+'/weights%Y%m%d%H%M%S.h5')
                os.makedirs(save_dir, exist_ok=True)
                agent.save_weights( fname )

    except KeyboardInterrupt:
        print('Catch KeyboardInterrupt')
    
    finally:
        fname = datetime.now().strftime(save_dir+'/weights_fin.h5')
        os.makedirs(save_dir, exist_ok=True)
        agent.save_weights( fname )
    if visualize: env.render(close=True)
    print('[end train] took : ', datetime.now()-train_start)

import pickle
def test(agent, env, nb_test, visualize=True):
    done = True
    episode=0
    step = 0
    log_reward = [ ]
    log_states0  = [ ]
    log_states1  = [ ]
    log_done  = [ ]

    for episode in range(1,nb_test+1):
        try:
            step = 0
            done = False
            state0 = env.reset()
            log_reward = [ ]
            log_states0  = [ ]
            log_states1  = [ ]
            log_done  = [ ]
            if visualize: env.render()

            while not done:
                action, evaluation_action, evaluation_state, alpha = agent.get_action(
                    state0, evalu=True, features=True,
                    attention_score = True
                )

                env.set_evaluation(evaluation_action, evaluation_state, alpha)
                state1, reward, done, _ = env.step(action)
                if visualize: env.render()

                agent.append(
                    state0, state1, reward, done, action
                )
                log_reward.append(reward)
                log_done.append(done)
                log_states0.append(state0)
                log_states1.append(state1)
                
                state0 = deepcopy(state1)

                if done:
                    print('episode : ', episode)
                    print(f'[steps:{len(log_done)}][reward {sum(log_reward)}]')
                    print()
        except KeyboardInterrupt:
            if not done:
                print('episode : ', episode)
                print(f'[steps:{len(log_done)}][reward {sum(log_reward)}]')
                print()
            if input('End test ? > [Y/N]\n').upper()=='Y':
                break
    if visualize: env.render(close=True)

def plot_log(file):
    try:
        import pandas as pd
        agent_num = 5
        cols = ['episode']
        for idx in range(agent_num):
            cols += [f'agent{idx:02d}_steps', f'agent{idx:02d}_reward', f'agent{idx:02d}_actor_loss', f'agent{idx:02d}_critic_loss']
        df = pd.read_csv(file, header=None, names=cols)

        episodes = df['episode'].values
        agent_num = int((len(df.columns)-1)//4)
        agents = []
        for idx in range(agent_num):
            agents.append(
                [
                    df[f'agent{idx:02d}_steps'].values,
                    df[f'agent{idx:02d}_reward'].values,
                    df[f'agent{idx:02d}_actor_loss'].values,
                    df[f'agent{idx:02d}_critic_loss'].values,
                ]
            )
    except ImportError:
        import csv
        with open(file,'r') as f:
            reader = csv.reader(f)
            
            for idx, row in enumerate(reader):
                if idx==0:
                    episodes = []
                    agents = [
                        [
                            [] for _ in range(4)
                        ] for _ in range(int((len(row)-1)//4))
                    ]
                else:
                    episodes.append(row[0])
                    for n in range(len(agents)):
                        agents[n][0].append( row[1+n*4] )
                        agents[n][1].append( row[2+n*4] )
                        agents[n][2].append( row[3+n*4] )
                        agents[n][3].append( row[4+n*4] )
    
    #####################################################################
    import os
    import shutil
    import matplotlib.pyplot as plt
    import mpl_toolkits.axes_grid1
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    cmap = plt.get_cmap('tab10')
    save_dir = datetime.now().strftime('%Y%m%d%H%M%SResults/')
    os.makedirs(save_dir)
    for agent_idx, data in enumerate(agents):
        fig, axs = plt.subplots(4,1,figsize=(10,8))
        
        for idx, (ax, label) in enumerate(zip(
            axs, [ 'nb_steps','reward', 'actor_loss', 'critic_loss']
        )):
            ax.scatter(
                episodes, data[idx], color=cmap(agent_idx/10), s=1
            )
            ax.set_ylabel(label)
            ax.grid()
        plt.subplots_adjust(hspace=0.5)
        fig.savefig(save_dir+f'agent{agent_idx:02d}.png')
        fig.suptitle(f'Results : Agent{agent_idx:02d}')
        plt.close()
    shutil.move(file, save_dir+file)
    print('save => ', save_dir)

if __name__=='__main__':
    plot_log('log.csv')