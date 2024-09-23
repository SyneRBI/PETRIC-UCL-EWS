# PETRIC: PET Rapid Image reconstruction Challenge

[![website](https://img.shields.io/badge/announcement-website-purple?logo=workplace&logoColor=white)](https://www.ccpsynerbi.ac.uk/events/petric/)
[![wiki](https://img.shields.io/badge/details-wiki-blue?logo=googledocs&logoColor=white)](https://github.com/SyneRBI/PETRIC/wiki)
[![register](https://img.shields.io/badge/participate-register-green?logo=ticktick&logoColor=white)](https://github.com/SyneRBI/PETRIC/issues/new/choose)
[![leaderboard](https://img.shields.io/badge/rankings-leaderboard-orange?logo=tensorflow&logoColor=white)](https://petric.tomography.stfc.ac.uk/leaderboard)
[![discord](https://img.shields.io/badge/chat-discord-blue?logo=discord&logoColor=white)](https://discord.gg/Ayd72Aa4ry)


## TAGS
- v0.0, BSREM with constant step size and the full gradient
- v0.1, BUG
- v0.2, BSREM with Barzilai-Borwein step size and the full gradient 
- v0.3 BUG
- v0.4 DOG-BSREM
- v0.5 Warm-Start SAGA

### Creating tags note:

Make sure on the right branch


```
git tag -a <tagname> -m '<message>'
git push origin --tags
```

## Approaches
To compute the number of subsets we use the functions in **utils/number_of_subsets.py**. This is a set of heuristic rules: 1) number of subsets has to be divisible by the number of views, 2) the number of subsets should have many prime factors (this results in a good herman meyer order), 3) I want at least 8 views in each subset and 4) I want at least 5 subsets. The function in **utils/number_of_subsets.py** is probably not really efficient. 

For TOF flight data, we do not use rule 3) and 4). If several number of subsets have the same number of prime factors, we take the larger number of subsets. 

All in all, this results in: 50 views (TOF) -> 25 subsets, 128 views -> 16 subsets and 252 views -> 28 views.


### 1) Educated Warm Start (main_EWS_SAGA.py)

To reduce the time required to reach the minimiser, we want to start closer to the minimiser. A better initialisation could reduce the number of steps an iterative algorithm needs and thus reduce the time. To this end, we employ a neural network, to learn a suitable initial image. The network is a (small) 3D convolutional neural network. It takes as input the OSEM image, the (preconditioned) gradient of the likelihood and the (preconditioned) gradient of the prior. As the preconditioner we use the usuall choice taken in BSREM.  

This has the obvious disadavantage that the could also do one epoch of SGD for the same cost. Maybe one epoch is SGD would actually result in a better initial value? I dont know. One could try a version only relying on the initial OSEM image. 

We start with a few epochs of SGD and then move to SAGA. We use the BSREM preconditioner. We make use of [DOG](https://arxiv.org/abs/2302.12022) for a parameter-free adaptive step size rule. Here, the step size is given by 

$$\lambda_k = \frac{\max_{i < k} \| x_i - x_\text{init} \|_2}{ \sqrt{\sum_{i < k} \Delta x_i^2}} $$

where $\Delta x_i$ is the full update, i.e., the preconditioned gradient, as iteration $i$. This deviates from the original DOP paper as they only have no preconditioner and use the sum of gradients in the denominator. 

### 2) Adam-SAGA (adaptive moment estimation)

Adam is a popular first order stochastic optimisation algorithm heavily used in deep learning. Here, we combine ADAM with the gradient averaging from SAGA. 

We start with a few epochs of SGD and then move to SAGA. 


### 3) Full Gradient Descent (main_Full_Gradient.py)

As a final submission, we do not make use of subsets at all. Using the full gradient goes against almost all empirical results, which show that the convergence can be speed by using subsets. Nonetheless, in our experiments we found that compared quite favourably to some other stochastic methods. Here, we use the Barzilai-Borwein step size rule. We use the BSREM preconditioner.


### Setup on Hydra

I setup the enviroment on hydra as follows:

```
docker run --rm -it -v /home/alexdenker/pet/data:/mnt/share/petric:ro -v .:/workdir -w /workdir --gpus all --user root synerbi/sirf:edge-gpu /bin/bash 

pip install git+https://github.com/TomographicImaging/Hackathon-000-Stochastic-QualityMetrics

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```


## Challenge information

### Layout

The organisers will import your submitted algorithm from `main.py` and then run & evaluate it.
Please create this file! See the example `main_*.py` files for inspiration.

[SIRF](https://github.com/SyneRBI/SIRF), [CIL](https://github.com/TomographicImaging/CIL), and CUDA are already installed (using [synerbi/sirf](https://github.com/synerbi/SIRF-SuperBuild/pkgs/container/sirf)).
Additional dependencies may be specified via `apt.txt`, `environment.yml`, and/or `requirements.txt`.

- (required) `main.py`: must define a `class Submission(cil.optimisation.algorithms.Algorithm)` and a list of `submission_callbacks`
- `apt.txt`: passed to `apt install`
- `environment.yml`: passed to `conda install`
- `requirements.txt`: passed to `pip install`

You can also find some example notebooks here which should help you with your development:
- https://github.com/SyneRBI/SIRF-Contribs/blob/master/src/notebooks/BSREM_illustration.ipynb

### Organiser setup

The organisers will execute (after downloading https://petric.tomography.stfc.ac.uk/data/ to `/path/to/data`):

```sh
docker run --rm -it -v /path/to/data:/mnt/share/petric:ro -v .:/workdir -w /workdir --gpus all synerbi/sirf:edge-gpu /bin/bash
# ... or ideally synerbi/sirf:latest-gpu after the next SIRF release!
pip install git+https://github.com/TomographicImaging/Hackathon-000-Stochastic-QualityMetrics
# ... conda/pip/apt install environment.yml/requirements.txt/apt.txt
python petric.py
```

> [!TIP]
> `petric.py` will effectively execute:
>
> ```python
> from main import Submission, submission_callbacks  # your submission
> from petric import data, metrics  # our data & evaluation
> assert issubclass(Submission, cil.optimisation.algorithms.Algorithm)
> Submission(data).run(numpy.inf, callbacks=metrics + submission_callbacks)
> ```

<!-- br -->

> [!WARNING]
> To avoid timing out (5 min runtime), please disable any debugging/plotting code before submitting!
> This includes removing any progress/logging from `submission_callbacks`.

- `data` to test/train your `Algorithm`s is available at https://petric.tomography.stfc.ac.uk/data/ and is likely to grow (more info to follow soon)
  + fewer datasets will be used by the organisers to provide a temporary [leaderboard](https://petric.tomography.stfc.ac.uk/leaderboard)
- `metrics` are calculated by `class QualityMetrics` within `petric.py`

Any modifications to `petric.py` are ignored.


## Team

Imraj Singh, Alexander Denker, Zeljko Kereta (University College London)