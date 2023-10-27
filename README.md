

# Reinforcement Learning for Stock Market Trading

This project is based on the research paper ["A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"](https://arxiv.org/abs/1706.10059). The goal is to use reinforcement learning to optimize stock market trading.

## Project Overview

The environment is represented by the prices of all orders throughout the market’s history up to the moment where the state is taken. Due to the vastness of the full order history, we sample the order-history information by dividing the time into periods and taking the maximum, minimum, and closing prices in each period.

The action is represented as: $\left(w_1, \ldots, w_n\right)$ 

The state is represented as: $\left(X_t, w_{t-1}\right)$ where prices are $X_t$
Reward function: $ R\left(\boldsymbol{s}_{1}, \boldsymbol{a}_{1}, \cdots, \boldsymbol{s}_{t_{\mathrm{f}}}, \boldsymbol{a}_{t_{\mathrm{f}}}, \boldsymbol{s}_{t_{\mathrm{f}}+1}\right) & :=\frac{1}{t_{\mathrm{f}}} \ln \frac{p_{\mathrm{f}}}{p_{0}}=\frac{1}{t_{\mathrm{f}}} \sum_{t=1}^{t_{\mathrm{f}}+1} r_{t} .$

Training procedure is Deterministic Policy Gradient with mini-batches.

Portfolio-Vector Memory stores the network outputs: stacks of portfolio vectors chronologically, initializes uniform weights before any network training. Each training step, a policy network loads the portfolio vector of the previous period from the memory location at t-1, and overwrites the memory at t with its output. During training the values in the memory converge.

## Data

The data for this project can be downloaded from the following links:

- [Input Data](https://drive.google.com/file/d/1VPPXoJEaZnE6NTY6h8GKJ3ZpCUmESppz/view?usp=sharing)
- [Individual Stocks Data for Analysis](https://drive.google.com/file/d/1k1Ad956uHQlJD-N9otFJ0vgPWhhxJCPw/view?usp=sharing) (Please unzip this file after downloading)

## Environment Setup

To set up your environment to run this project, please use pip to install the necessary requirements:

```
pip install -r requirements.txt
```

## Running an Experiment

To start an experiment, use the following command:

```
python experiment.py
```

To start an experiment with analysis, use the following command:

```
python experiment.py --vis-analysis True
```

## Results

![image](https://github.com/AlicePH/RL_project/blob/main/assets/train/portfolio_weights_episode_1.png)
![image](https://github.com/AlicePH/RL_project/blob/main/assets/analysis/portfolio_value.png)

You can check the results of your experiments in the `assets` folder.

