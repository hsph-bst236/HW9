# Homework 9: Reinforcement Learning

The due date is April 24 at midnight. Please follow the [code squad rules](https://junwei-lu.github.io/bst236/chapter_syllabus/syllabus/#code-squad). If you are using the late days, please note in the head of README.md that "We used XX late days this time, and we have XX days remaining". 

The main purpose of this homework is to help you:

- Implement the Reinforcement Learning algorithms.
- Experience the RL training process.
- Understand the training losses curves and metrics for RL algorithms.
- Practice the gym environment.



In this homework, you need to fill in the missing code under `#TODO` in the python files in the `src` folder.  We will give specific instructions for each part. 

We suggest you to test your code on CPU first. If it runs correctly, then you can try to tune the hyperparameters, train the model, and test the model on the GPUs on the class cluster.

This homework involves hours of large model training  (probably longer than 24 hours to finish all the training but you can stop at any time you believe the training is enough for you are up to time for deadline). Please plan your time accordingly. 

## Problem: Train Your Own Super Mario Bros 

You will implement reinforcement learning algorithms to train a policy to play NES Super Mario Bros.

The complete training process to make your Mario complete World 1-1 might take around one day on the class cluster. So you do not have to complete the training. As the program will save model and replay the game every few iterations, you can stop at any time you believe the training is enough for you are up to time for deadline. Still, due to long training time, we still suggest you to plan your time accordingly. Below is our training result for reference.

![Training Result](./demo.gif)

Your training result might not be as good as the above result. Do not push too hard to your Mario. It is already a big achievement for him to jump over the first Kuribo.



As a general guideline, you can start with the code in `mario_play.ipynb` to explore the environment. You can see that the state space is a 3D tensor representing 4 stacked frames `(num_stacked_frames, height, width) = (4, 13, 16)`. The 4 frames are the most recent 4 frames, with each 2 frames separated by 3 frames. The tiles, pipes, Mario, and enemies are represented by different numbers.

![Mario State Space](./obs.png)

For each part of the problem below, please read the comments and input/output format in the function/class definition carefully to understand the requirements. For each part, you can find the part you need to fill in `# TODO: [part a]` in the code.

We use `wandb` to log the training process. You can find the loss curves and the performance of the trained model in `wandb`. A tip is to [smooth the loss curves](https://docs.wandb.ai/guides/app/features/panels/line-plot/smoothing/) to make them easier to observe.

You need to work the following parts:

### Part a: Implement the Policy and Value Networks

In this part, you will policy network and value networks. Your task is to complete the `MlpPolicy` class in `src/model.py`. This network consists of two components: a policy network that outputs a distribution over actions, and a value network that estimates the value function. Both networks should share the same architecture with two hidden layers of size 64 and `nn.ReLU` activations. 

Note that the two networks do not share parameters though they have the same architecture and we implement them in the same class. 

Please read the comments and input/output format in the `MlpPolicy` class. Be careful with the input and output dimensions. The input to the networks is a 3D tensor representing stacked frames `(num_stacked_frames, height, width)`, which needs to be flattened before processing. 
For example,the `forward` function should return a tuple of two elements: the first element is a `Categorical` distribution over possible actions, and the second element is the value function in the shape of `(batch_size, 1)`.

### Part b: Implement the Advantage Function

In this part, you will implement the code to calculate the reward-to-go and the generalized advantage estimate (GAE) in `src/ppo/gae.py`.

We describe the vanilla policy gradient algorithm here for you to match code with the equations:

- Initialize policy parameters $\theta$ randomly
- For k = 1, 2, ... do:
    - Generate $K=$`n_workers` trajectories $\mathcal{D}_k = \{\tau_i| i = 1, 2, ..., K \}$ using policy $\pi_{\theta_k}$, each trajectory $\tau_i$ has $T=$`worker_steps` steps
    - Calculate reward-to-go $\hat{R}_t$ for each trajectory
    - **Actor**: Minimize the loss: $\theta_k = \arg \min_{\theta} \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} - \log \pi_{\theta_k}(a_t |s_t) \hat{R}_t$


Note $d_i$ is 0/1 value representing whether the game is done at the $i$-th step. Here is the formula for Reward-to-go: 

$$
\hat{R}_t = \sum_{i=t}^{T} \gamma^{i-t} (1-d_i)r_i
$$

Here is the formula for GAE: 

$$
{A}_t = \delta_t + \gamma\lambda\delta_{t+1} + \ldots + (\gamma\lambda)^{T-t+1}\delta_{T-1}
$$

where $\delta_t = (1-d_t) r_t + \gamma V(s_{t+1}) - V(s_t)$ or recursively ${A}_t = \delta_t + \gamma\lambda{A}_{t+1}$. $\lambda$ is the GAE parameter which is `lambda_` in code.

You need to implement the `reward_to_go` function and the `GAE` class in `src/ppo/gae.py`. You can first read the two `sample` functions in `src/train_ppo.py` and `src/train_vpg.py` to understand how the input data is generated.

Note that $d_i$ is the `done` boolean `np.ndarray` representing whether the game is done at the $i$-th step. `n_workers` is the number of parallel games running in the environment for you to generate $K$ independent trajectories in the algorithm above and `worker_steps` is the number of steps $T$ for each game. You can read `sample` in `src/train_ppo.py` and `src/train_vpg.py` to understand how the samples are generated.

### Part c: Implement the Vanilla Policy Gradient Algorithm

In this part, you will implement the above vanilla policy gradient algorithm in `src/train_vpg.py`.

You need to implement the `_calc_loss` method in `src/train_vpg.py` to calculate the policy gradient loss $- \log \pi_{\theta_k}(a_t |s_t) \hat{R}_t$ averaged over a mini-batch of samples. See the `train` method in `src/train_vpg.py` for more details.

Once you fill in the above parts, you can test your code by running `python src/train_vpg.py` for a smaller number of updates.


Once you have finished the above parts, you can train the model by submitting the `python src/run_vpg.py` on the class cluster. You can explore the hyperparameters in `configs` to see how they affect the training, but the current settings in the file should give you a reasonable performance though vanilla policy gradient algorithm might not give you a good performance.

**Tip**: You can submit multiple jobs with different hyperparameters configuration to the class cluster. We suggest you explore try both `learning_rate = 3e-4` and a linear scheduler `learning_rate = linear_schedule(3e-4)` which might stabilize the training. But do not submit too many jobs (<4 each part) to block the available resources for other squads.

You may need to apply for 24 hours of GPU time on the class cluster to finish the training. And use the `wandb` to monitor the training process. However, the vanilla policy gradient algorithm may not give you a good performance of training so you can stop the training earlier if time is limited. You can check the model output in the `results` folder.

### Part d: Implement the Proximal Policy Optimization Loss

In this part, you will implement the PPO loss in `src/ppo/__init__.py`.
We again describe the PPO algorithm here for you to match code with the equations:

- Initialize policy parameters $\theta$ randomly
- For k = 1, 2, ... do:
    - Generate trajectories $\mathcal{D}_k = \{\tau_i\}$ using policy $\pi_{\theta_k}$
    - Compute advantage $A_t$ by Generalized Advantage Estimation (GAE) using `GAE` class in `src/ppo/gae.py`
    - Update policy and value networks: $\theta_{k+1}, \phi_{k+1} = \arg \min_{\theta, \phi} \sum_{\tau \in \mathcal{D}_k}  \sum_{t=0}^{T} \left[ L(s_t,a_t) \right]$


In our implementation, you need to implement three loss functions:

First, the policy loss:

$$
L_\text{PPO}(\theta_k,\theta) = \min\Bigl( \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_k}(a_t | s_t)} A_t, \text{clip}\Bigl(\frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_k}(a_t | s_t)}, 1-\epsilon, 1+\epsilon\Bigr) A_t \Bigr)
$$

and $\text{clip}$ is the clipping function:

$$
\text{clip}(x, a, b) = \max\Bigl(a, \min(x, b)\Bigr)
$$

Second, the value function loss is the least square loss:

$$
L_{VF}(\phi) = \frac{1}{2} \mathbb{E}[(V_{\phi}(s_t) - \hat{R}_t)^2]
$$

Finally, the total loss is:

$$
 L(s_t,a_t) = -L_\text{PPO}(\theta_k,\theta) + c_1 L_{VF}(\phi_k,\phi) - c_2  H(\pi_{\theta}(s)),
$$


where $H(\pi_{\theta}(s))$ is the entropy of the policy. In `src/train_ppo.py`, $c_1$ is `value_loss_coef`, $c_2$ is `entropy_bonus_coef`, and `clip_range` is $\epsilon$.

You need to implement the `forward` method in `src/ppo/__init__.py` to calculate the PPO loss. And assemble all the losses to calculate the total loss in `src/train_ppo.py`. You can first read the two `sample` functions in `src/train_ppo.py` to understand how the input data is generated.



### Part e: Train the PPO Algorithm

In this part, you will train the PPO algorithm in `src/train_ppo.py`.

Once you finish the above parts, you can test your code by running `python src/train_ppo.py` for a smaller number of updates.


Then you need to submit the `python src/run_ppo.py` on the class cluster. You can explore the hyperparameters in `configs` to see how they affect the training, but the current settings in the file should give you a reasonable performance.

**Tip**: You can submit multiple jobs with different hyperparameters configuration to the class cluster. We suggest you try both `learning_rate = 3e-4` and a linear scheduler `learning_rate = linear_schedule(3e-4)` which might stabilize the training. But do not submit too many jobs (<4 each part) to block the available resources for other squads.

Monitor the training process using `wandb`. Model will save the checkpoint and replay the game every `save_freq` updates in the folder `results`. If you find the episode reward is close to 3000, that means your Mario can pass World 1-1. From our implementation, it takes around 10 million steps to pass World 1-1 with around 24 hours of training time. 
However, due to minor implementation differences, your performance may vary from ours. So you can decide when to stop the training based on your left time, the convergence of the training process, and saved models performance.

### Part f: Report

In this part, you need to write a report to  summarize the results and compare the performance of the PPO algorithm and the vanilla policy gradient algorithm.

List the loss curves and the performance of the trained model in `wandb` and explain how each curve might help you understand the training process.



















