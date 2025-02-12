import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import clone_model, Model


"""
現状actorの行動の選択肢は1
"""
class Actor(tf.keras.Model):
    def __init__(self, wp_shape, oth_shape, nb_action, mode='sum'):
        super().__init__()
        self.nb_features = 512
        self.mode=mode
        self.flatten = Flatten()
        self.dropout = Dropout(0.1)
        self.leaky_relu = LeakyReLU(0.1)

        self.hidden_wp_1 = Dense(256, activation='linear',kernel_initializer='he_normal')
        self.hidden_wp_2 = Dense(256, activation='linear',kernel_initializer='he_normal')
        self.hidden_wp_3 = Dense(128, activation='linear', kernel_initializer='glorot_normal')

        self.hidden_oth_1 = Dense(256, activation='linear',kernel_initializer='he_normal')
        self.hidden_oth_2 = Dense(256, activation='linear',kernel_initializer='he_normal')
        self.hidden_oth_3 = Dense(128, activation='linear', kernel_initializer='glorot_normal')


        self.softsign = Activation('softsign')
        # self.tanh = Activation('tanh')

        self.hidden_actor_1 = Dense(256, activation='linear',kernel_initializer='he_normal')
        self.hidden_actor_2 = Dense(256, activation='linear',kernel_initializer='he_normal')
        self.hidden_actor_3 = Dense(128, activation='linear',kernel_initializer='he_normal')
        self.hidden_actor_4 = Dense(nb_action, activation='softsign',kernel_initializer='glorot_normal')

        # self.attention1 = Dense(128, activation='linear',kernel_initializer='he_normal')
        # self.attention2 = Dense(128, activation='linear',kernel_initializer='he_normal')
        # self.attention3 = Dense(1, activation='linear',kernel_initializer='he_normal')

        self.__call__(
            np.random.random((1,)+wp_shape),
            np.random.random((1,2,)+oth_shape),
        )

    def call(self, input_wp, input_oth=None, training=False):
        if type(input_wp) != list:
            input_tensor = True
            input_wp = [input_wp]
            input_oth = [input_oth]
        else:
            input_tensor = False
        out = []
        for idx in range(len(input_wp)):
            wp = self.leaky_relu(self.hidden_wp_1(self.flatten(input_wp[idx]), training=training), training=training)
            wp = self.leaky_relu(self.hidden_wp_2(self.dropout(wp, training=training), training=training), training=training)
            wp = self.hidden_wp_3(self.dropout(wp, training=training), training=training)
            
            if input_oth[idx] is not None:
                oth = self.leaky_relu(self.hidden_oth_1(input_oth[idx], training=training), training=training)
                oth = self.leaky_relu(self.hidden_oth_2(self.dropout(oth, training=training), training=training), training=training)
                oth = self.hidden_oth_3(self.dropout(oth, training=training), training=training)
                _out = self.leaky_relu( wp + tf.reduce_sum(oth, axis=1 ) )
            else:
                _out = self.leaky_relu( wp )

            _out = self.leaky_relu(self.hidden_actor_1(self.dropout(_out, training=training), training=training), training=training)
            _out = self.leaky_relu(self.hidden_actor_2(self.dropout(_out, training=training), training=training), training=training)
            _out = self.leaky_relu(self.hidden_actor_3(self.dropout(_out, training=training), training=training), training=training)
            _out = self.hidden_actor_4(_out, training=training)
            out.append(_out)
        
        if input_tensor:
            return out[0]
        else:
            return out


class Critic(tf.keras.Model):
    def __init__(self, wp_shape, oth_shape, nb_action, mode='sum'):
        super().__init__()
        self.mode=mode    
        self.flatten = Flatten()
        self.dropout = Dropout(0.1)

        self.hidden_wp_1 = Dense(256, activation='relu',kernel_initializer='he_normal')
        self.hidden_wp_2 = Dense(256, activation='relu',kernel_initializer='he_normal')
        self.hidden_wp_3 = Dense(128, activation='relu',kernel_initializer='he_normal')
        self.hidden_wp_4 = Dense(  1, activation='linear', kernel_initializer='glorot_normal' )

        self.hidden_oth_1 = Dense(256, activation='relu',kernel_initializer='he_normal')
        self.hidden_oth_2 = Dense(256, activation='relu',kernel_initializer='he_normal')
        self.hidden_oth_3 = Dense(128, activation='relu',kernel_initializer='he_normal')
        self.hidden_oth_4 = Dense(  1, activation='linear',kernel_initializer='glorot_normal')

        self.__call__(
            np.random.random((2,nb_action)),
            np.random.random((2,)+wp_shape),
            np.random.random((2,2,)+oth_shape),
        )



    def call(self, action_input, input_wp, input_oth=None, training=False):
        if type(input_wp) != list:
            input_tensor = True
            input_wp = [input_wp]
            input_oth = [input_oth]
            action_input = [action_input]
        else:
            input_tensor = False
        out = []
        _len=0
        for idx in range(len(input_wp)):
            wp = self.hidden_wp_1(
                tf.concat([self.flatten(input_wp[idx]), action_input[idx]], axis=1),
                training=training
            )
            wp = self.hidden_wp_2(self.dropout(wp, training=training), training=training)
            wp = self.hidden_wp_3(self.dropout(wp, training=training), training=training)
            wp = self.hidden_wp_4(wp, training=training)
            
            if input_oth[idx] is not None:
                oth = self.hidden_oth_1(
                    tf.concat(
                        [
                            input_oth[idx],
                            tf.tile(
                                tf.expand_dims(action_input[idx], -1),
                                tf.constant([1,input_oth[idx].shape[1],1,]),
                            ),
                        ], axis=-1
                    ),
                    training=training
                )
                oth = self.hidden_oth_2(self.dropout(oth, training=training), training=training)
                oth = self.hidden_oth_3(self.dropout(oth, training=training), training=training)
                oth = self.hidden_oth_4(oth, training=training)
                oth = tf.reduce_sum(oth, axis=1)
                _out = wp + oth
            else:
                _out = wp

            out.append(_out)
        return tf.concat(out, axis=0)
    
    def evalu(self, action_tensor, wp_tensor, oth_tensor=None):
        
        wp = self.hidden_wp_1(
            tf.concat([self.flatten(wp_tensor), action_tensor], axis=1),
            training=False
        )
        wp = self.hidden_wp_2(self.dropout(wp, training=False), training=False)
        wp = self.hidden_wp_3(self.dropout(wp, training=False), training=False)
        wp = self.hidden_wp_4(wp, training=False)
        
        if oth_tensor is not None:
            oth = self.hidden_oth_1(
                tf.concat(
                    [
                        oth_tensor,
                        tf.tile(
                            tf.expand_dims(action_tensor, -1),
                            tf.constant([1,oth_tensor.shape[1],1,]),
                        ),
                    ], axis=-1
                ),
                training=False
            )
            oth = self.hidden_oth_2(self.dropout(oth, training=False), training=False)
            oth = self.hidden_oth_3(self.dropout(oth, training=False), training=False)
            oth = self.hidden_oth_4(oth, training=False)
            return tf.concat([wp, tf.squeeze(oth, [-1])], axis=1)
        else:
            return wp

def MLP(wp_shape, oth_shape, act_shape, nb_features=512):
    # encoder
    actor = Actor(wp_shape, oth_shape, len(act_shape))
    critic = Critic(wp_shape, oth_shape, len(act_shape))

    return actor, critic

class Memory():
    def __init__(self, maxlen, window_length):
        self.window_length = window_length
        self.state0 = deque(maxlen=maxlen)
        self.state1 = deque(maxlen=maxlen)
        self.reward = deque(maxlen=maxlen)
        self.done = deque(maxlen=maxlen)
        self.action = deque(maxlen=maxlen)

    def append(self, s, s_next, r, d, a):
        if self.window_length is None:
            self.state0.append(s)
            self.state1.append(s_next)
            self.reward.append(r)
            self.done.append(d)
            self.action.append(a)
        else:
            if len(self.done)>=1 and not self.done[-1]:
                state0 = list(self.state0[-1][1:]) + [s]
                state1 = list(self.state1[-1][1:]) + [s_next]
            else:
                # print('zeros_like')
                state0 = [np.zeros_like(s) for _ in range(self.window_length-1)] + [s]
                state1 = [np.zeros_like(s) for _ in range(self.window_length-2)] + [s,s_next]
            
            self.state0.append(np.array(state0))
            self.state1.append(np.array(state1))
            self.reward.append(r)
            self.done.append(d)
            self.action.append(a)

    def get_recent_obs(self, obs):
        if self.window_length is None:
            return obs
        state = [obs]
        if len(self.done)>=1 and not self.done[-1]:
            # print('reuse')
            # print(self.state0[-1])
            state = list(self.state0[-1][1:]) + state
        else:
            # print('zeros_like')
            state = [np.zeros_like(obs) for _ in range(self.window_length-1)] + state
        state = np.array(state)
        # print(state.shape)
        return state
    
    def get_sample_idx(self, num):
        idxs = np.random.randint(
            self.window_length-1, len(self.state0), num
        )
        for i in range(num):
            while self.done[idxs[i]-1] and self.done[idxs[i]]:
                idxs[i] =  np.random.randint(
                    self.window_length-1,
                    len(self.state0),
                )
        return idxs
    
    def __getitem__(self, idx):
        if type(idx) is int:
            return [self.state0[idx], self.state1[idx], self.reward[idx], self.done[idx], self.action[idx]]
        else:
            return [
                [self.state0[i], self.state1[i], self.reward[i], self.done[i], self.action[i]]
                for i in idx
            ]

class OrnsteinUhlenbeckProcess():
    def __init__(self, theta, mu=0., sigma=1., dt=1e-3, size=1, sigma_min=None, n_steps_annealing=1000, nb_wormup=0):
        assert n_steps_annealing>nb_wormup
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.size = size

        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0
        self.nb_wormup = nb_wormup

        if sigma_min is not None:
            self.sigma_del = -float(sigma - sigma_min) / float(n_steps_annealing - nb_wormup)
            self.sigma_ini = sigma
            self.sigma_min = sigma_min
        else:
            self.sigma_del = 0.
            self.sigma_ini = sigma
            self.sigma_min = sigma

        self.noise_que = deque(maxlen=1000)

        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        self.noise_que.append(x)
        return x

    def reset_states(self):
        self.noise_que = deque(maxlen=1000)
        self.x_prev = np.random.normal(self.mu,self.current_sigma,self.size)

    @property
    def current_sigma(self):
        if self.n_steps<self.nb_wormup:
            sigma = self.sigma_ini
        else:
            sigma = max(self.sigma_min, self.sigma_del * float(self.n_steps-self.nb_wormup) + self.sigma_ini)
        return sigma

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

@tf.function
def train_step(model, _ins, _ture, loss_func, optimizer):
    with tf.GradientTape() as tape:
        preds = model(_ins)
        loss = loss_func(_ture, preds)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

class Agent():
    def __init__(self,
        wp_shape, oth_shape, act_shape,
        ##
        random, memory,
        actor_optimizer, critic_optimizer,
        batch_size, gamma,
        target_model_update,
        training=True,
        ):
        self.random = random
        self.memory = memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_model_update = 0.001
        self.grad_cliping = None
        self.training = training
        self.target_model_update = target_model_update
        # 暫定的な設定
        self.action_num = 1

        # 学習に使うmdel
        self.actor, self.critic = MLP(
            wp_shape, oth_shape, act_shape
        )

        # 予測に使うmodel
        self.actor_target, self.critic_target = MLP(
            wp_shape, oth_shape, act_shape
        )

        # 学習用の設定
        self.critic_loss = tf.keras.losses.Huber(delta=1.0)
        self.critic_opt = critic_optimizer
        
        self.actor_opt = actor_optimizer
        assert self.critic_opt!=self.actor_opt


        self.grad_info = [[],[]]
        for name in self.actor.trainable_weights:
            self.grad_info[0] += ['(Actor)'+name.name+'_min', '(Actor)'+name.name+'_max', '(Actor)'+name.name+'_mean']
            self.grad_info[1] += [None, None, None,] 
        for name in self.critic.trainable_weights:
            self.grad_info[0] += ['(Critic)'+name.name+'_min', '(Critic)'+name.name+'_max', '(Critic)'+name.name+'_mean']
            self.grad_info[1] += [None, None, None,]

    def train(self, train_actor=True, trian_critic=True, smoothing_gain=0):
        sample_idx = self.memory.get_sample_idx(self.batch_size)
        
        exp = self.memory[sample_idx]
        # state0 = [exp[n][0] for n in range(self.batch_size)]
        # state1 = [exp[n][1] for n in range(self.batch_size)]
        # rewards = [[exp[n][2]] for n in range(self.batch_size)]
        # done = [[exp[n][3]] for n in range(self.batch_size)]
        # action = [[exp[n][4]] for n in range(self.batch_size)]
        
        shape2idx = []
        state0 = []
        state1 = []
        rewards = []
        done = []
        action = []
        oth_input_num = 10
        for n in range(self.batch_size):
            shape = np.array(exp[n][0]).shape
            if shape not in shape2idx:
                shape2idx.append(shape)
                state0.append([])
                state1.append([])
                rewards.append([])
                done.append([])
                action.append([])
            idx = shape2idx.index(shape)
            if len(state0[idx])==0:
                oth_num = int( (shape[1]-3)/oth_input_num )
                state0[idx].append(
                    [exp[n][0][:,:3]]
                )
                state1[idx].append(
                    [exp[n][1][:,:3]]
                )
                if oth_num>0:
                    state0[idx].append(
                        [
                            [
                                exp[n][0][:,3+oth_idx*oth_input_num:3+(1+oth_idx)*oth_input_num].flatten()
                                for oth_idx in range(oth_num)
                            ]
                        ]
                    )
                    state1[idx].append(
                        [
                            [
                                exp[n][1][:,3+oth_idx*oth_input_num:3+(1+oth_idx)*oth_input_num].flatten()
                                for oth_idx in range(oth_num)
                            ]
                        ]
                    )
                else:
                    state0[idx].append(None)
                    state1[idx].append(None)
            else:
                oth_num = int( (shape[1]-3)/oth_input_num )
                state0[idx][0].append( exp[n][0][:,:3] )
                state1[idx][0].append( exp[n][1][:,:3] )
                if oth_num>0:
                    state0[idx][1].append(
                        [
                            exp[n][0][:,3+oth_idx*oth_input_num:3+(1+oth_idx)*oth_input_num].flatten()
                            for oth_idx in range(oth_num)
                        ]
                    )
                    state1[idx][1].append(
                        [
                            exp[n][1][:,3+oth_idx*oth_input_num:3+(1+oth_idx)*oth_input_num].flatten()
                            for oth_idx in range(oth_num)
                        ]
                    )
            rewards[idx].append(exp[n][2])
            done[idx].append(exp[n][3])
            action[idx].append(exp[n][4])

        state0 = [ [tf.convert_to_tensor(np.array(_s), dtype=np.float32) if _s is not None else None for _s in s ] for s in state0]
        state1 = [ [tf.convert_to_tensor(np.array(_s), dtype=np.float32) if _s is not None else None for _s in s ] for s in state1]
        rewards = [tf.convert_to_tensor(np.array(r).reshape((len(r),1)), dtype=np.float32) for r in rewards]
        action0 = [tf.convert_to_tensor(np.array(a).reshape((len(a),1)), dtype=np.float32) for a in action]
        # Trueが1なので，0になるように引き算
        done = [tf.convert_to_tensor((1 - np.array(d)).reshape((len(d),1)), dtype=np.float32) for d in done]
        
        action1 = [
            tf.convert_to_tensor(
                a.numpy().reshape((int(a.numpy().size/self.action_num),self.action_num)),
                dtype=np.float32
            )
            for a in  self.actor_target(
                [ s[0] for s in state1],
                [ s[1] for s in state1],
            )
        ]
        
        losses = self.update(
            state0, state1, rewards, done, action0, action1,
            train_actor=train_actor, trian_critic=trian_critic, smoothing_gain=smoothing_gain
        )


        self.update_target(self.target_model_update)

        return losses

    @tf.function        
    def update_target(self, tau):
        update_target(
            self.actor_target.trainable_weights,
            self.actor.trainable_weights,
            tau
        )
        update_target(
            self.critic_target.trainable_weights,
            self.critic.trainable_weights,
            tau
        )
    
    def update(self, state0, state1, reward, done, actions0, actions1, train_actor=True, trian_critic=True, smoothing_gain=0):
        #update critic
        reward_concat = tf.concat(reward, axis=0)
        done_concat = tf.concat(done, axis=0)
        actions0_concat = tf.concat(actions0, axis=0)
        actions0_concat_noised = actions0_concat
        actions1_concat = tf.concat(actions1, axis=0)

        state0_input_wp = [ s[0] for s in state0 ]
        state0_input_oth = [ s[1] for s in state0 ]
        
        state1_input_wp = [ s[0] for s in state1 ]
        state1_input_oth = [ s[1] for s in state1 ]
        
        state0_input_wp_noise  = [ s[0] + tf.random.normal(s[0].shape,0,0.02) if s[0] is not None else None for s in state0  ]
        state0_input_oth_noise = [ s[1] + tf.random.normal(s[1].shape,0,0.02) if s[1] is not None else None for s in state0  ]
        actions0_noised = [ _a + tf.random.normal(_a.shape,0,0.02) for _a in actions0]
        # state1_input_wp_noise = [ s[0] + tf.random.normal(s[0].shape,0,0.05) for s in state1 ]
        # state1_input_oth_noise = [ s[1] + tf.random.normal(s[1].shape,0,0.05) for s in state1 ]
        
        with tf.GradientTape() as tape:
            target_reward = tf.stop_gradient(
                reward_concat + done_concat*self.gamma*self.critic_target(actions1, state1_input_wp, state1_input_oth, training=False)
            )
            critic_loss = self.critic_loss(
                self.critic(
                    # actions0,
                    actions0_noised,
                    state0_input_wp,
                    state0_input_oth,
                    training=True,
                ),
                target_reward
            )

            # l2 normalize
            loss_reg_critic = 0
            for var in self.critic.trainable_variables:
                if 'bias' not in var.name:
                    loss_reg_critic += tf.reduce_mean(tf.square(var))
            
            critic_loss += loss_reg_critic*1e-4

            Lt = tf.reduce_mean(
                tf.square(
                    tf.stop_gradient(
                        self.critic(
                            actions0,
                            state0_input_wp,
                            state0_input_oth,
                            training=True,
                        )
                    ) - self.critic(
                        actions0_noised,
                        state0_input_wp,
                        state0_input_oth,
                        training=True,
                    )
                )
            ) + tf.reduce_mean(
                tf.square(
                    tf.stop_gradient(
                        self.critic(
                            actions0,
                            state0_input_wp,
                            state0_input_oth,
                            training=True,
                        )
                    ) - self.critic(
                        actions1,
                        state1_input_wp,
                        state1_input_oth,
                        training=True,
                    )
                )
            )
            critic_loss += Lt*1e-2

        critic_grad = tape.gradient(
            critic_loss, self.critic.trainable_variables
        )
        if self.grad_cliping is not None:
            critic_grad = [
                None if gradient is None else tf.clip_by_norm(gradient, self.grad_clipping)
                for gradient in critic_grad
            ]
        if trian_critic:
            self.critic_opt.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables)
            )
        
    
        #update actor
        with tf.GradientTape() as tape:
            # j_pi
            actor_out = self.actor(
                state0_input_wp,
                state0_input_oth,
                training=True
            )
            pred = self.critic(
                actor_out,
                state0_input_wp,
                state0_input_oth,
                training=False
            )
            loss_j_pi = -tf.reduce_mean(pred)

            # l2 normalize
            loss_reg_actor = 0
            for var in self.actor.trainable_variables:
                if 'bias' not in var.name:
                    loss_reg_actor += tf.reduce_mean(tf.square(var))
            
            actor_loss = loss_j_pi + loss_reg_actor*1e-3

            # smoothing
            next_ = self.actor(
                state1_input_wp,
                state1_input_oth,
            )
            noised_ = self.actor(
                state0_input_wp_noise,
                state0_input_oth_noise,
            )
            
            if type(actor_out) is list:
                Lt = tf.sqrt(
                    tf.reduce_mean(
                        tf.square(
                            tf.stop_gradient(tf.concat(actor_out, axis=0))
                            - tf.concat(next_, axis=0)
                        )
                    )
                ) + tf.sqrt(
                    tf.reduce_mean(
                        tf.square(
                            tf.stop_gradient(tf.concat(actor_out, axis=0))
                            - tf.concat(noised_, axis=0)
                        )
                    )
                )    
            else:
                Lt = tf.sqrt(
                    tf.reduce_mean(
                        tf.square(
                            tf.stop_gradient(actor_out) - next_
                        )
                    )
                ) + tf.sqrt(
                    tf.reduce_mean(
                        tf.square(
                            tf.stop_gradient(actor_out) - noised_
                        )
                    )
                )
            
            actor_loss += Lt*1e-2

        actor_grad = tape.gradient(
            actor_loss, self.actor.trainable_variables
        )
        if self.grad_cliping is not None:
            actor_grad = [
                None if gradient is None else tf.clip_by_norm(gradient, self.grad_clipping)
                for gradient in actor_grad
            ]
        if train_actor:
            self.actor_opt.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables)
            )


        self.grad_info = [[],[]]
        for name, grad in zip(self.actor.trainable_weights, actor_grad):
            self.grad_info[0] += ['(Actor)'+name.name+'_min', '(Actor)'+name.name+'_max', '(Actor)'+name.name+'_mean']
            self.grad_info[1] += [grad.numpy().min(), grad.numpy().max(), grad.numpy().mean()]
        for name, grad in zip(self.critic.trainable_weights, critic_grad):
            self.grad_info[0] += ['(Critic)'+name.name+'_min', '(Critic)'+name.name+'_max', '(Critic)'+name.name+'_mean']
            self.grad_info[1] += [grad.numpy().min(), grad.numpy().max(), grad.numpy().mean()]

        return actor_loss, critic_loss

    def append(self, s, s_next, r, d, a):
        self.memory.append(
            s, s_next, r, d, a
        )

    def get_action(self, state, evalu=False, features=False):
        _in = self.memory.get_recent_obs(state)
        # act = self.actor.predict( [np.array([_in])] ).flatten()[0]
        oth_input_num = 10
        wp = np.array(_in)[:,:3]
        if int((_in.shape[1]-3)/oth_input_num) > 0:
            oth = np.array(
                [
                    np.array(_in)[:,3+oth_input_num*n:3+oth_input_num*(n+1)].flatten()
                    for n in range(int((_in.shape[1]-3)/oth_input_num))
                ]
            )
        else:
            oth = None

        act = self.actor(
            np.array([wp]),
            np.array([oth]) if oth is not None else None,
        ).numpy()[0]

        if self.training:
            noise = self.random.sample()
            act += noise
        act = np.clip(act, -1, 1)
        if evalu:
            evalu = self.critic.evalu(
                np.array([act]),
                np.array([wp]),
                np.array([oth]) if oth is not None else None,
            ).numpy()[0]

            state_actions = np.array(
                [
                    self.critic.evalu(
                        np.array([np.full(act.shape,_a)]),
                        np.array([wp]),
                        np.array([oth]) if oth is not None else None,
                    ).numpy()[0] for _a in np.linspace(-1,1,21)
                ]
            )
            return act, evalu, state_actions

        else:
            return act
    
    def load_weights(self, fname):
        self.actor.load_weights(
            fname.replace('.', '_actor.')
        )
        self.critic.load_weights(
            fname.replace('.', '_critic.')
        )
    
    def save_weights(self, fname):
        self.actor.save_weights(
            fname.replace('.', '_actor.')
        )
        self.critic.save_weights(
            fname.replace('.', '_critic.')
        )


if __name__=='__main__':
    nb_trainstep = 100000
    # # random = OrnsteinUhlenbeckProcess(
    # #     size=1,
    # #     theta=.01,
    # #     mu=0.,
    # #     dt=15.0,
    # #     sigma=0.10,
    # #     sigma_min=0.0010,
    # #     n_steps_annealing=int(nb_trainstep)
    # # )
    # random = OrnsteinUhlenbeckProcess(
    #     size=1,
    #     theta=.01,
    #     mu=0.,
    #     dt=30.0,
    #     # dt=5.0,
    #     sigma=0.12,
    #     sigma_min=0.010,
    #     n_steps_annealing=int(nb_trainstep*0.8),
    #     nb_wormup = int(nb_trainstep*0.2)
    # )


    # x = range(nb_trainstep)
    # y = [ random.sample() for _ in x]
    # import os
    # import time
    # import numpy as np
    # from glob import glob
    # import matplotlib.pyplot as plt
    # import mpl_toolkits.axes_grid1
    # # divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    # # cax = divider.append_axes('right', '5%', pad='3%')
    
    # from datetime import datetime    
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 10
    # plt.rcParams['figure.dpi'] = 100
    # plt.rcParams['mathtext.fontset'] = 'cm'
    # plt.rcParams['axes.axisbelow'] = True
    # plt.rcParams['xtick.direction'] = 'in'
    # plt.rcParams['ytick.direction'] = 'in'


    # fig, ax = plt.subplots(1,1,figsize=(10,4))
    # # ax.plot(x,y)
    # ax.scatter(x,y,s=0.1)
    # ax.axvline(x=int(nb_trainstep*0.8), ymin=0, ymax=1, color='gray', linestyle='dashed')
    # ax.axvline(x=int(nb_trainstep*0.2), ymin=0, ymax=1, color='gray', linestyle='dashed')
    # ax.axvline(x=nb_trainstep, ymin=0, ymax=1, color='gray', linestyle='dashed')
    
    # ax.set_xlim(-100,1000)
    # plt.show()

    wp_shape = (5,4)
    oth_shape = (50,)
    nb_action = 1
    critic = Critic(wp_shape, oth_shape, nb_action, nb_features=512, mode='sum')


    a = np.random.random((2,nb_action))
    b = np.random.random((2,)+wp_shape)
    # c = np.random.random((2,2,)+oth_shape)
    c = None
    out = critic(a, b, c)
    print(out, end='\n\n')

    evalu = critic.evalu(a, b, c)
    print(evalu)

    b = np.random.random((2,)+wp_shape)
    c = np.random.random((2,2,)+oth_shape)
    actor = Actor(wp_shape, oth_shape, nb_action, nb_features=512, mode='sum')
    print(actor(input_wp=b, input_oth=c))