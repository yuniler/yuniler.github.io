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

