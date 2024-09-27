# PETRIC: PET Rapid Image reconstruction Challenge

[![website](https://img.shields.io/badge/announcement-website-purple?logo=workplace&logoColor=white)](https://www.ccpsynerbi.ac.uk/events/petric/)
[![wiki](https://img.shields.io/badge/details-wiki-blue?logo=googledocs&logoColor=white)](https://github.com/SyneRBI/PETRIC/wiki)
[![register](https://img.shields.io/badge/participate-register-green?logo=ticktick&logoColor=white)](https://github.com/SyneRBI/PETRIC/issues/new/choose)
[![leaderboard](https://img.shields.io/badge/rankings-leaderboard-orange?logo=tensorflow&logoColor=white)](https://petric.tomography.stfc.ac.uk/leaderboard)
[![discord](https://img.shields.io/badge/chat-discord-blue?logo=discord&logoColor=white)](https://discord.gg/Ayd72Aa4ry)


## Reconstruction Methods - Educated Warm Start
To reduce the time required to reach the minimiser, we want to start closer to the minimiser. A better initialisation could reduce the number of steps an iterative algorithm needs and thus reduce the time. To this end, we employ a neural network, to learn a suitable initial image. The network is a (small) 3D convolutional neural network. All layers in the network have no bias and we ReLU activation functions. This results in a 1-homogeneous network. In this way, the network should be independent of the intensity of the image. It takes as input only the OSEM image, provided by the organisers. The network weights are available in the folder *checkpoint/*.

We employ three different classical iterative algortihms.

### 1) BSREM preconditioner, DOwG step size rule, SAGA gradient estimation (in branch: main)
We employ the traditional BSREM algorithm. The number of subsets is automatically calculated using some heuristics (see below). We use [DoWG](https://arxiv.org/abs/2305.16284) (Distance over Weighted Gradients) for the step size calculation. We use SAGA to get an estimate of the full gradient.


### 2) BSREM preconditioner, DOwG step size rule, SGD (in branch: ews_sgd)
We employ the traditional BSREM algorithm. The number of subsets is automatically calculated using some heuristics (see below). We use [DoWG](https://arxiv.org/abs/2305.16284) (Distance over Weighted Gradients) for the step size calculation.

### 3) adaptive preconditioner, full gradient descent, Barzilai-Borwein step size rule (in branch: full_gd)
The characteristics of the datasets varied a lot, i.e., we had low count data, different scanner setups, TOF data, and so on. We tried a lot of different algorithms and approaches, both classical and deep learning based, but it was hard to design a method, which works consistently for all these different settings. Based on this experience, we submit a full gradient descent algorithm with a Barzilai-Borwein step size rule. Using the full gradient goes against almost all empirical results, which show that the convergence can be speed by using subsets. However, most work look at a speed up with respect to number of iterations. For the challenge, we are interested in raw computation time. With respect to raw computation time, we sometimes saw only a minor different between full gradient descent and gradient descent using subsets.  

At the start of the optimisation we compare the norm of the gradient of the RDP prior with the norm of the gradient of the full objective function. If this fraction is less than 0.5, we simply use the BSREM type preconditioner. If this fraction is larger than 0.5, we use a similar preconditioner to [Tsai et al. (2018)](https://pubmed.ncbi.nlm.nih.gov/29610077/). However, we do not update the Hessian row sum of the likelihood term. Further, we found that the Hessian row sum of the RDP prior was instable, so instead we only used the diagonal elements of the Hessian evaluated at the current iterate. This defines a kind of strange preconditioner, where only the RDP part in the preconditioner is updated.


### Subset Choice 
To compute the number of subsets we use the functions in **utils/number_of_subsets.py**. This is a set of heuristic rules: 1) number of subsets has to be divisible by the number of views, 2) the number of subsets should have many prime factors (this results in a good herman meyer order), 3) I want at least 8 views in each subset and 4) I want at least 5 subsets. The function in **utils/number_of_subsets.py** is probably not really efficient. 

For TOF flight data, we do not use rule 3) and 4). If several number of subsets have the same number of prime factors, we take the larger number of subsets. 

All in all, this results in: 50 views (TOF) -> 25 subsets, 128 views -> 16 subsets and 252 views -> 28 views.


## Challenge information

### Creating tags note:

Make sure on the right branch


```
git tag -a <tagname> -m '<message>'
git push origin --tags
```


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

We thank Zeljko Kereta for valuable discussion.

- Imraj Singh and Alexander Denker (University College London)

