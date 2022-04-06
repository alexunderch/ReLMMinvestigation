import numpy as np
import matplotlib.pyplot as plt
from typing import List
def make_plot(files: List[str], labels: List[str], title: str) -> None:
    """Visualizer"""
    fig = plt.figure()
    dicts = [np.load(file_, allow_pickle= True) for file_ in files]
    for ind, dict_ in enumerate(dicts):
        steps, quals = [], []
        for object_ in dict_:
            obj_ = dict(object_) 
            steps.append(obj_['num_samples_total'])  
            quals.append(obj_['eval_success_ratio'])  
        plt.plot(steps, quals, label = labels[ind])
    plt.title("Strategies comparison")
    plt.xlabel("#Steps")
    plt.ylabel("Eval success ratio")
    plt.legend(loc = 'best')
    return fig


def main():
    fig = make_plot(files = [
                        "/home/alex_ch/ReLMM/others/results/perturb/std_uncertainty.npy",
                        "/home/alex_ch/ReLMM/others/results/perturb/20objects.npy",
                      ],
               labels = ["10 objects", "20 objects"], 
               title = "Pretrain Impact")
    fig.savefig('first_fig.png')


if __name__ == '__main__': main()