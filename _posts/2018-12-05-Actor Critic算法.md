---
layout:     post
title:      Actor Critic算法
subtitle:   算法详解
date:       2018-12-05
author:     Hybot
header-img: img/post-md.jpg
catalog: true
tags:
    - AI
    - DRL
    - Actor-Critic
---

最近要做一个连续动作输出的深度学习网络模型，如何去实现？100%都会告诉你A3C，这个中文名为异步演员批评家优势算法名字巨长，不过基本原理是Actor-Critic，那么我们从AC开始，代码实现是来自莫烦老师的GitHub

[](https://morvanzhou.github.io/tutorials/machinelearning/reinforcement-learning/)

# 1.Actor-Critic算法

先抛出结论，AC是value-based算法Q-Learning和model-based算法Policy Gradient算法的结合，为什么要结合，莫烦老师讲解的比较明白：

> 我们有了像 Q-learning这么伟大的算法, 为什么还要瞎折腾出一个 Actor-Critic? 原来 Actor-Critic 的 Actor 的前生是 Policy Gradients, 这能让它毫不费力地在连续动作中选取合适的动作, 而 Q-learning 做这件事会瘫痪. 那为什么不直接用 Policy Gradients 呢? 原来 Actor Critic 中的 Critic 的前生是 Q-learning 或者其他的 以值为基础的学习法 , 能进行单步更新, 而传统的 Policy Gradients 则是回合更新, 这降低了学习效率.

所以我们使用AC的目的就十分明显了。一、我们需要连续action输出，二、我们需要单步更新网络来提高学习效率。

* **From Q-Learning to AC**

为了进一步理解AC，我们从演变角度来看看AC的实现原理：

QL有两个部分，

1. 我们有Q(s, a)表，用来表示在s状态下执行a动作后得到的期望收益Q；

2. 为了让算法的策略搜索有创新性，使用e-greedy来选择action，大于e值才执行Q值最大的aciton

第一部分就是Critic，第二部分就是Actor，我试图使用这种方式来解释AC里面没有e-greedy的原因，所以AC算法如何探索随机策略，我还是心存疑惑的，加个todo。

* **From PG to AC**

PG就是Policy Gradient的缩写啊，虽然算法用到的不多，我们还是说明一下PG的原理，

我们有一个pi(s, a)表示状态s下执行动作a的概率，在每一次学习中，我们就follow当前policy一条路走到黑，看看最后的return是多少。如果return不错，那么把pi(s,a)的值稍微加大一点，鼓励下次继续选择动作a。但是每当现有policy稍微一迭代更新，在s状态采取动作a后，依照这个policy开始一条路走到黑最后的return就变化，那么就会导致我们一下子鼓励a，一下子不鼓励a，不利于训练（尤其是用神经网络等非线性模型作为Function Approximator的时候）。一个改进思路是干脆我们把return的平均值记下来，取代之前用的单次simulation的return，作为s状态采取动作a到底好不好的度量。好处是return的平均值（近似Q(s,a)）变化就慢多了，起码正负号稳定多了。这里用return平均值近似的Q(s,a)就是Critic。前面的pi(s,a)就是actor。

知道了AC的基本原理，我们看一下AC算法流程图：

![](https://github.com/hybug/hybug.github.io/blob/master/img/AC-pic1.png?raw=true)

# 2.代码解析

我们一步一步实现，先看Actor网络

## 2.1 Actor

了解一个网络，第一步了解网络的输入和输出，我们需要state来输出action，我们需要action和td_error来计算loss值。

···
self.s = tf.placeholder(tf.float32, [1, n_features], "state")
self.a = tf.placeholder(tf.float32, None, name="act")
self.td_error = tf.placeholder(tf.float32, None, name="td_error")  # TD_error
···

* **Actor网络实现**

我们使用的gym中的那个让棒子保持平衡的游戏，游戏很简单，所以使用的是全连接层，然而我的需求是肯定需要使用CNN的，理论上CNN应该也没问题，但是得试试。

···
l1 = tf.layers.dense(
    inputs=self.s,
    units=30,  # number of hidden units
    activation=tf.nn.relu,
    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
    bias_initializer=tf.constant_initializer(0.1),  # biases
    name='l1'
)

mu = tf.layers.dense(
    inputs=l1,
    units=1,  # number of hidden units
    activation=tf.nn.tanh,
    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
    bias_initializer=tf.constant_initializer(0.1),  # biases
    name='mu'
)

sigma = tf.layers.dense(
    inputs=l1,
    units=1,  # output units
    activation=tf.nn.softplus,  # get action probabilities
    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
    bias_initializer=tf.constant_initializer(1.),  # biases
    name='sigma'
)
···

如何让我们的全连接层网络输出连续的action值呢，使用了tf.distribution.normal函数，具体方法另一篇笔记中记录了。

···
with tf.name_scope('exp_v'):
    log_prob = self.normal_dist.log_prob(self.a)  # loss without advantage
    self.exp_v = log_prob * self.td_error  # advantage (TD_error) guided loss
    # Add cross entropy cost to encourage exploration
    self.exp_v += 0.01*self.normal_dist.entropy()

with tf.name_scope('train'):
    self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v, global_step)    # min(v) = max(-v)
···

**loss = -log(prob)*ttd_error**

最后得到exp_v 和 train_op， 小问题需要注意，PG中算法迭代是逐步增加pi的值，我们需要是逐步增加exp_v，max(v) = min(-v)

* **Action的选择**

···
def choose_action(self, s):
    s = s[np.newaxis, :]
    return self.sess.run(self.action, {self.s: s})  # get probabilities for all actions
···

* **Actor的训练**

输入state，action，ed_errror，得到我们需要的值

···
def learn(self, s, a, td):
    s = s[np.newaxis, :]
    feed_dict = {self.s: s, self.a: a, self.td_error: td}
    _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
    return exp_v
···

## 2.2 Critic网络

Critic要反馈给Actor一个时间差分值，来决定Actor选择动作的好坏，如果时间差分值大的话，说明当前Actor选择的这个动作的惊喜度较高，需要更多的出现来使得时间差分值减小。时间差分计算公式：

**TD = r + gamma*v_(s_) - v(s)**

v表示s输入到critic网络中得到的Value值/Q值，我们需要输入s得到v；s_得到v_；v，v_和r得到TD

* **Critic网络结构**

···
with tf.variable_scope('Critic'):
    l1 = tf.layers.dense(
        inputs=self.s,
        units=30,  # number of hidden units
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
        bias_initializer=tf.constant_initializer(0.1),  # biases
        name='l1'
    )

    self.v = tf.layers.dense(
        inputs=l1,
        units=1,  # output units
        activation=None,
        kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
        bias_initializer=tf.constant_initializer(0.1),  # biases
        name='V'
    )
    with tf.variable_scope('squared_TD_error'):
    self.td_error = tf.reduce_mean(self.r + GAMMA * self.v_ - self.v)
    self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
with tf.variable_scope('train'):
    self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
···

* **TD计算和训练**

···
def learn(self, s, r, s_):
    s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

    v_ = self.sess.run(self.v, {self.s: s_})
    td_error, _ = self.sess.run([self.td_error, self.train_op],
                                      {self.s: s, self.v_: v_, self.r: r})
    return td_error
···

## 2.3 AC网络训练

···
for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    ep_rs = []
    while True:
        # if RENDER:
        env.render()
        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)
        r /= 10

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1
        ep_rs.append(r)
···

# 3.参考资料

[](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-1-actor-critic/)
