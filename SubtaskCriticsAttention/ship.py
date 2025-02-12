import numpy as np
from copy import deepcopy
"""
野本の1次近似のKTモデル
・最も単純な操縦運動モデル
・舵角と回頭角速度の関係を表す
・複雑な動きを解くときはMMGモデルでないと動きが解けない

r_dush = (K*lambda - r)/T
K,T : Zig-Zag実験から決まる定数
r   : 回頭速度
lambda:舵角


@propatyメソッドはself.dataで呼び出せる
stepのretuenはdef data(self)で定義したもの
"""
class KTmodel():
    def __init__(self, K=0.1827, T=11.10):
        # 各パラメータの定義
        self._K = K
        self._T = T
        self._cal_dt = 1
        self._X = None
        self._Y = None
        self._Sog = None
        self._Heading = None
        self._Rot = None
        self._Loa = None
        self._rudder=0

    def reset(self, x_init, y_init, sog_init, head_init, rot_init):
        # 初期化(位置，速度，船首方位，回頭角)
        self._X = x_init
        self._Y = y_init
        self._Sog = sog_init
        self._Heading = head_init
        self._Rot = rot_init
        self._ruuder = 0

        return self.data
    
    def step(self, rudder, dt):
        """
        dt[sec]だけ船を進める
        while文でself._cal_dtごとに分割して計算する
        """
        assert self._X is not None, "Not initilized !!"
        self._rudder = rudder
        cal_time = dt/np.ceil(dt/self._cal_dt) # self._cal_dtごとに分割するときの1stepの時間
        for _ in range(int(np.ceil(dt/self._cal_dt))):
            r_next = self._Rot + ( ( self._K*rudder - self._Rot )/self._T )*cal_time
            head_next = self._Heading + (r_next + self._Rot)*cal_time
            self._X += self._Sog*np.sin( np.radians( (head_next + self._Heading)/2 ) )*cal_time
            self._Y += self._Sog*np.cos( np.radians( (head_next + self._Heading)/2 ) )*cal_time
            self._Rot = r_next
            self._Heading = head_next

        return self.data
    
    @property
    def data(self):
        return deepcopy([ self._X, self._Y, self._Sog, self._Heading, self._Rot])

if __name__=='__main__':
    ship = KTmodel()

    own_log = [ ship.reset(0,0,10*1.852/3600, 0, 0) ]

    sim_time = 1200
    rudder = 1
    for _ in range(sim_time):
        if _ >= 0:
            rudder = -35
        else:
            rudder = 0
        own_log.append( ship.step(rudder, 1) )

    own_log = np.array(own_log).T

    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    #plt.xkcd()import os

    fig, ax = plt.subplots(1, 1, figsize=(8,8))

    ax.plot(
        own_log[0], own_log[1]
    )
    
    wide = max(
        [ max(own_log[0])-min(own_log[0]), max(own_log[1])-min(own_log[1]) ]
    )
    ax.set_xlim( (max(own_log[0])+min(own_log[0]))*0.5 - wide*0.6, (max(own_log[0])+min(own_log[0]))*0.5 + wide*0.6 )
    ax.set_ylim( (max(own_log[1])+min(own_log[1]))*0.5 - wide*0.6, (max(own_log[1])+min(own_log[1]))*0.5 + wide*0.6 )
    ax.grid()
    plt.show()