# Learning 3-opt heuristics for traveling salesman problem via deep reinforcement learning

Code accompanying the paper [Learning 3-opt heuristics for traveling salesman problem via
deep reinforcement learning](https://proceedings.mlr.press/v157/sui21a.html).


## Dependencies
python 3.8.10

Torch

Numpy

tqdm

Apex

pyconcorde

Matplotlib


## Training
Train the model using:
```
python PGTSP20.py
python PGTSP50_100.py

```

## Testing
Evaluate the model using:
```
python TestLearnedAgent.py --load_path <model directory>/<model name>.pt --n_points <number of nodes> --test_size <number of instances in testset> --render

```

## Citation
If you this code is useful in your research, please cite our paper:
```
@inproceedings{sui2021learning,
  title={Learning 3-opt heuristics for traveling salesman problem via deep reinforcement learning},
  author={Sui, Jingyan and Ding, Shizhe and Liu, Ruizhi and Xu, Liming and Bu, Dongbo},
  booktitle={Asian Conference on Machine Learning},
  pages={1301--1316},
  year={2021},
  organization={PMLR}
}
```

This project extends the foundations established by the previous study "Learning 2-opt Heuristics for the TSP via Deep Reinforcement Learning".
