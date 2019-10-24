# Cross Entropy Method

## Usage

```bash
$ python cem.py cartpole --num_process 6 --epochs 8 --batch_size 4096

$ python cem.py pendulum --num_process 6 --epochs 15 --batch_size 4096
```

Cross Entropy Method (CEM) is a gradient free optimization algorithm that fits parameters by iteratively resampling from an elite population.  The model learns only from a single scalar (total episode reward).

```python
for epoch in num_epochs:
  sampling a population from a distribution
  testing that population using the environment
  selecting the elites (judged by total episode reward)
  refitting the sampling distribution (to the elites)
```

CEM is easily parallelizable - this code base runs large batches across multiple processes, making it very efficient in clock time.

The total number of episodes run in an experiment is given by:

```python
num_episoes = num_epochs * num_processes * batch_size
```

## Setup

```bash
$ git clone https://github.com/ADGEfficiency/cem

$ cd cem

$ pip install -r requirements.txt
```

The two dependencies of this project are `Open AI gym` and `matplotlib`.

## Results

Results for the gym environments `CartPole-v0` and `Pendulum-v0`.

### Cartpole 

![](assets/cartpole.png)

![](assets/cartpole.gif)

### Pendulum 

![](assets/pendulum.png)

![](assets/pendulum.gif)
