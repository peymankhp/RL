# Dynamic Programming

The term dynamic programming (DP) refers to a collection of algorithms that can be used to compute optimal policies **given a perfect model of the environment as a Markov decision process (MDP)**. Classical DP algorithms are of **limited utility** in reinforcement learning, both because of their assumption of a **perfect model** and because of their **great computational expense**, but they are still important theoretically, providing an essential foundation for the understanding of the other methods. In fact, all of these methods can be viewed as attempts to achieve much the same effect as DP, only with less computation and without assuming a perfect model of the environment. 

**The key idea of DP, and of reinforcement learning generally, is the use of value functions to organize and structure the search for good policies.**

## Policy Evaluation (Prediction)

First, we consider how to compute the [state-value function v*π*](https://www.notion.so/Summary-Reinforcement-Learning-An-Introduction-2nd-Edition-Richard-Sutton-and-Andrew-Barto-1fb8e9353d4a80419b75e076b8b2fb7c?pvs=21) for an arbitrary policy *π*. This is called policy evaluation in the DP literature. We also refer to it as the prediction problem. The existence and uniqueness of v*π* are guaranteed as long as either γ < 1 or eventual termination is guaranteed from all states under the policy *π*.