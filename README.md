## Hybrid Robotic Grasping with a Soft Multimodal Gripper and a Deep Multistage Learning Scheme

[[Paper]](https://arxiv.org/pdf/2202.12796.pdf)

[Fukang Liu](https://fukangl.github.io/)<sup>1,2</sup>,[Fuchun Sun](https://scholar.google.com/citations?user=DbviELoAAAAJ&hl=en)<sup>2</sup>,[Bin Fang](https://scholar.google.com/citations?user=5G47IcIAAAAJ&hl=en)<sup>2</sup>,[Xiang Li](https://scholar.google.com/citations?user=6EIX-JQAAAAJ&hl=en)<sup>2</sup>,Songyu Sun<sup>3</sup>,[Huaping Liu](https://scholar.google.com/citations?user=HXnkIkwAAAAJ&hl=en)<sup>2</sup><br/>

<sup>1</sup>Carnegie Mellon University </br> 
<sup>2</sup>Tsinghua University </br>
<sup>3</sup> University of California, Los Angeles </br> 

## Installation

The implementation requires the following dependencies:

* [Python](https://www.python.org/), [PyTorch](https://pytorch.org/), [NumPy](https://numpy.org/), [OpenCV-Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html), [SciPy](https://scipy.org/), [Matplotlib](https://matplotlib.org/), [CoppeliaSim](https://www.coppeliarobotics.com/)</br> 


## Instructions

1. Checkout this repository and download the [datasets](https://github.com/fukangl/SMG-multimodal-grasping/blob/main/datasets-objects.zip).

1. Run CoppeliaSim (navigate to your CoppeliaSim directory and run `./coppeliaSim.sh`). From the main menu, select `File` > `Open scene...`, and open the file `code/simulation/simulation-lc.ttt`(lightly-cluttered) or `simulation-hc.ttt`(highly-cluttered) from this repository. Choose the `Vortex` physics engine for simulation (you can also choose other physics engines that CoppeliaSim supports (e.g., `Bullet`, `ODE`), but `Vortex` works best for the simulation model of SMG).

1. In another terminal window, run the following (example).

```shell
python main.py --is_sim --is_cluttered --explore_rate_decay
```


## Training

To train an Reactive Enveloping and Sucking Policy (E+S Reactive) in simulation with lightly cluttered environment, run the following:

```shell
python main.py --is_sim --method 'reactive' --explore_rate_decay
```

To train an Reactive Enveloping, Sucking and Enveloping_then_Sucking Policy (E+S+ES Reactive) in simulation with lightly cluttered environment, run the following:

```shell
python main.py --is_sim --method 'reactive' --is_ets --explore_rate_decay
```

To train a DRL Enveloping and Sucking Policy (E+S DRL) in simulation with lightly cluttered environment, run the following:

```shell
python main.py --is_sim --method 'reinforcement' --explore_rate_decay
```

To train a DRL multimodal grasping policy (E+S+ES DRL(PE+OO)) in simulation with lightly cluttered environment, run the following:

```shell
python main.py --is_sim --method 'reinforcement' --is_ets --explore_rate_decay
```

## Evaluation

To test your own pre-trained model, simply change the location of `--snapshot_file`. For example, for testing the pre-trained E+S+ES DRL(PE+OO) model in simulation with lightly cluttered environment, run the following:

```shell
python main.py --is_sim --method 'reinforcement' --is_ets --explore_rate_decay \
--is_testing \
--load_snapshot --snapshot_file 'YOUR-SNAPSHOT-FILE-HERE'
```

## Bibtex
If you find this code useful, please cite:

```
@misc{SMG2022,
    title={Hybrid Robotic Grasping with a Soft Multimodal Gripper and a Deep Multistage Learning Scheme},
    author={Liu, Fukang and Fang, Bin and Sun, Fuchun and 
    Li, Xiang and Sun, Songyu and Liu, Huaping},
    journal={arXiv preprint arXiv:2202.12796},
    year={2022}
}
```

## Acknowledgements
This code was developed using [visual-pushing-grasping](https://github.com/andyzeng/visual-pushing-grasping).
