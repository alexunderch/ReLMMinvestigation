from ast import Call
import os, sys
import argparse
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List

from discretizer import Discretizer
from envs import GraspingEnv
import policies as plc
import samplers as smpl
from replay_buffer import ReplayBuffer
import train_functions as trf
from training_loop import training_loop
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import Callable
from functools import partial
import copy
def make_plot(results: List[dict], labels: List[str], title: str) -> None:
    """Visualizer"""
    fig = plt.figure()
    for ind, dict_ in enumerate(results):
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


@dataclass
class GraspingExperiment(object):
    training_config: dict
    sampler_config: Callable

    environment: GraspingEnv
    eval_environment: GraspingEnv

    sampler: Callable
    eval_sampler: Callable

    training_function: Callable
    train_buffer: ReplayBuffer

    validation_function: Callable = None
    validation_buffer: ReplayBuffer = None

    def setup(self, logits_model: Callable):
        """Init function""" 
        self.logits_models = [logits_model for _ in range(self.training_config['num_models'])]
        self.optimizers = [tf.optimizers.Adam(learning_rate=3e-4) for _ in range(self.training_config['num_models'])]
        
        self.environment.reset()
        self.eval_environment.reset()

        tr = [
                self.training_function(logits_model, optimizer, np.prod(self.training_config['discrete_dimensions']))
                                       for logits_model, optimizer in zip(self.logits_models, self.optimizers)
             ]

        self.validation_function = partial(self.validation_function, 
                                          discrete_dimension = self.training_config['discrete_dimensions'],
                                          logits_model = self.logits_models)
        self.train_ = lambda data: [train(data) for train in tr]

        sampler_kwargs = self.sampler_config(True)
        sampler_kwargs.update({"logits_models": self.logits_models, "env": self.environment})
        self.sampler = self.sampler(**sampler_kwargs)

        eval_sampler_kwargs = self.sampler_config(False)
        eval_sampler_kwargs.update({"logits_models": self.logits_models,
                                     "env": self.eval_environment,
                                     "min_samples_before_train": 0})
        self.eval_sampler =  self.eval_sampler(**eval_sampler_kwargs)
        return self
    def run_experiment(self, loop_config: dict, savedir: str) -> dict:
        """main class function"""   
        print("before loop")
        all_diagnostics, train_buffer, validation_buffer = training_loop(
                                                        env=self.environment, eval_env=self.eval_environment,
                                                        sampler=self.sampler, eval_sampler=self.eval_sampler,
                                                        train_buffer=self.train_buffer, validation_buffer=self.validation_buffer,
                                                        train_function=self.train_, validation_function=None,
                                                        **loop_config
                                                    )
        if savedir: np.save(os.path.join(savedir, "diagnostics"), all_diagnostics)
        return all_diagnostics



def train_grasp(args):
    savedir = "."
    trainloop_hyperparams = {
        'num_samples_per_env': 4,
        'num_samples_per_epoch': 100,
        'num_samples_total': args.num_samples_total,
        'min_samples_before_train': 1000,
        'num_eval_samples_per_epoch': 50,
        'train_frequency': 1,
        'num_train_repeat': 1,
        'train_batch_size': 128,
        'validation_prob': -1,
        'validation_batch_size': 128,
        'pretrain': args.pretrain
    }
    training_hyperparams = {
        'image_size': 60,
        'discrete_dimensions': [15, 15, 5] if args.use_theta else [15, 15],
        'num_models': 6
    }

    room_hyperparams ={
        "room_name": "simple",
        "num_objects_min": 10,
        "num_objects_max": 10,
        "use_theta": args.use_theta
    }

    # sampling_hyperparams = lambda x: {
    #     'alpha': 10.0,
    #     'beta': 10.0 if x else 0.0,
    #     'discretizer': Discretizer(sizes = training_hyperparams['discrete_dimensions'], 
    #                                mins = [0.3, -0.08], maxs = [0.4666666, 0.08]),
    #     'min_samples_before_train': trainloop_hyperparams['min_samples_before_train'],
    #     'deterministic': x,
    #     'aggregate_func': "mean",
    #     'uncertainty_func': "std" if x else None
    # }


   
    plot_data = []
    labels = []

    if args.name == "uncertainties": labels = [None, 'diff', 'std']
    print("LABELS:", labels)
    for label in labels:
        env = GraspingEnv(renders = args.render_train, rand_color = args.rand_color_train, 
                          rand_floor = args.rand_floor_train, **room_hyperparams)
        eval_env = GraspingEnv(renders = args.render_eval, rand_color = args.rand_color_eval, 
                              rand_floor = args.rand_floor_eval, **room_hyperparams)

        sampler = smpl.create_grasping_env_soft_q_sampler
        eval_sampler = smpl.create_grasping_env_soft_q_sampler

        train_buffer = ReplayBuffer(size = trainloop_hyperparams['num_samples_total'],
                                    observation_shape = (training_hyperparams['image_size'], training_hyperparams['image_size'], 3), 
                                    action_dim = 1, raw_action_dim = (2,))
        validation_buffer = ReplayBuffer(size = trainloop_hyperparams['num_samples_total'], 
                                        observation_shape = (training_hyperparams['image_size'], training_hyperparams['image_size'], 3), 
                                        action_dim = 1, raw_action_dim = (2,)) 

              
        sampling_hyperparams = lambda x: {
                                          'alpha': 10.0,
                                          'beta': 10.0 if x else 0.0,
                                          'discretizer': Discretizer(sizes = training_hyperparams['discrete_dimensions'], 
                                                                    mins = [0.3, -0.08], maxs = [0.4666666, 0.08]),
                                          'min_samples_before_train': trainloop_hyperparams['min_samples_before_train'],
                                          'deterministic': x,
                                          'aggregate_func': "mean",
                                          'uncertainty_func': label if x else None
                                      }
        experiment = GraspingExperiment(training_config=training_hyperparams, sampler_config = sampling_hyperparams,
                                      environment=env, eval_environment=eval_env,
                                      sampler=sampler, eval_sampler=eval_sampler,
                                      training_function=trf.create_train_discrete_Q_sigmoid, train_buffer=train_buffer,
                                      validation_function=trf.validation_discrete_sigmoid, validation_buffer=validation_buffer)

        experiment = experiment.setup(logits_model = plc.build_discrete_Q_model(image_size = training_hyperparams['image_size'], 
                                                                              discrete_dimension = np.prod(training_hyperparams['discrete_dimensions']),
                                                                              discrete_hidden_layers = [512, 512]))
        plot_data.append(experiment.run_experiment(loop_config=trainloop_hyperparams, savedir=None))

    fig = make_plot(plot_data,
                    labels = [str(label) for label in labels],
                    title = args.name)
    fig.savefig(f"{args.name}.png")
    plt.show()
    


    def curriculum(): pass

def main(args): pass

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="name of experiment", type=str, default='')
    parser.add_argument("--load_data", help="whether to preload data", default=False, action="store_true")
    parser.add_argument("--use_theta", help="whether to use a grasp angle", default=False, action="store_true")
    parser.add_argument("--rand_color_train", help="whether to randomize training object colors", default=True, action="store_true")
    parser.add_argument("--rand_floor_train", help="whether to randomize training pos and floors", default=True, action="store_true")
    parser.add_argument("--rand_color_eval", help="whether to randomize eval object colors", default=True, action="store_true")
    parser.add_argument("--rand_floor_eval", help="whether to randomize eval pos and floors", default=True, action="store_true")
    parser.add_argument("--policy", help="name of policy", type=str, default='soft_q')

    parser.add_argument("--render_train", help="whether to render training env", default=False, action="store_true")
    parser.add_argument("--render_eval", help="whether to render eval env", default=False, action="store_true")

    parser.add_argument("--pretrain", help="number of steps to pretrain for", type=int, default=0)
    parser.add_argument("--num_samples_total", help="number of samples total", type=int, default=int(100))

    
    args = parser.parse_args()
    train_grasp(args)
