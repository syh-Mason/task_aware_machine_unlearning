"""
contains functions for
1. fit the linear model using sklearn and analytical method
2. unlearn the linear model based on mse loss
3. unlearn the linear model based on cost loss
"""

from torch import nn
import torch
import numpy as np
import sys
sys.path.append(".")
import copy
from pathlib import Path
from utils.optimization import Operator


def load_influences(path, expected_size):
    """Load scores only after the final training/sensitive dataset is known."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Influence scores missing: {path}. Run gen_index.py with the same model, data settings and influence_dir first.')
    scores = np.load(path, allow_pickle=False)
    if scores.shape != (expected_size,) or not np.isfinite(scores).all():
        raise ValueError(f'Influence scores {path} must be finite with shape ({expected_size},); got {scores.shape}. Regenerate with matching data settings.')
    return torch.from_numpy(scores).float()

class AnalyticSolverAffine:
    """
    functions to train the affine model
    """
    def __init__(self, dataset_collect):
        for key in dataset_collect.keys():
            # flatten the feature and target
            new_dataset = copy.deepcopy(dataset_collect[key])
            new_dataset.feature = new_dataset.feature.view(-1, new_dataset.feature.shape[-1])
            new_dataset.target = new_dataset.target
            setattr(self, key, new_dataset)
    
    def select_dataset(self,mode):
        if mode == 'train':
            dataset = self.train
        elif mode == 'test':
            dataset = self.test
        elif mode == 'remain':
            dataset = self.remain
        elif mode == 'unlearn':
            dataset = self.unlearn
        
        # assert dataset.is_scale == True, "dataset should be scaled for affine map"
        # include the bias term into design matrix
        self.X = np.concatenate((dataset.feature.numpy(), np.ones((dataset.feature.shape[0], 1))), axis = 1)
        self.Y = dataset.target.numpy().flatten()
        
        # print('shape of X: {} Y: {}'.format(self.X.shape, self.Y.shape))
    
    def fit_regressor(self, mode):
        self.select_dataset(mode)
        
        A = np.matmul(self.X.T, self.X)
        # print(A)
        b = np.matmul(self.X.T, self.Y)
        parameter = np.linalg.solve(A, b)
        
        return parameter.T
    
    def cal_grad(self, mode, parameter):
        self.select_dataset(mode)
        A = np.matmul(self.X.T, self.X)
        b = np.matmul(self.X.T, self.Y)
        grad = 2 * (A@parameter - b) / len(self.Y)
        print('shape of Y: {} grad: {}'.format(self.Y.shape, grad.shape))
        return grad 
    
    def cal_hessian(self, mode, parameter):
        self.select_dataset(mode)
        hessian = 2 * self.X.T @ self.X / len(self.Y)
        return hessian

class AnalyticSolverNNAffine(AnalyticSolverAffine):
    """
    functions to train the affine model upon the neural network
    """
            
    def select_dataset(self,mode):
        if mode == 'train':
            dataset = self.train
        elif mode == 'test':
            dataset = self.test
        elif mode == 'remain':
            dataset = self.remain
        elif mode == 'unlearn':
            dataset = self.unlearn
        X = np.concatenate((dataset.feature.numpy(), np.ones((dataset.feature.shape[0], 1))), axis = 1)
        # X = dataset.feature.numpy()
        Y = dataset.target.numpy()
        
        # diagnolization
        no_output = Y.shape[1]
        self.X = np.kron(np.eye(no_output), X)
        self.Y = Y.T.flatten() # ! the target is flattened

class ModelAffine(nn.Module):
    """
    convert the affine model into a pytorch model
    """
    def __init__(self, parameter):
        super().__init__()
        weight = parameter[:-1]
        bias = parameter[-1]
        self.fc = nn.Linear(len(weight), 1, bias=True)
        self.fc.weight = nn.Parameter(torch.tensor(weight).view(1,-1).float())
        self.fc.bias = nn.Parameter(torch.tensor(bias).float())
    
    def forward(self,x):
        return self.fc(x).squeeze(axis=-1)

class ModelNNAffine(nn.Module):
    """
    convert the affine model upon the neural network into a pytorch model
    """
    
    def __init__(self, parameter, no_out = 14):
        super().__init__()
        parameter = parameter.reshape(no_out, -1)
        weight = parameter[:, :-1]
        bias = parameter[:, -1]
        self.fc = nn.Linear(weight.shape[1], weight.shape[0], bias=True)
        self.fc.weight = nn.Parameter(torch.tensor(weight).float())
        self.fc.bias = nn.Parameter(torch.tensor(bias).float())
    
    def forward(self,x):
        return self.fc(x)

def evaluate(model, dataset_collection, loss, case_config = None, 
            is_average = True, with_mispatch = False, is_save = False, save_dir = None):    
    """
    dataset_collection: a dictionary of dataset
    loss: 'mse' or 'mape' or 'cost'
    with_mismatch: if true, record the mismatch between the generation and load
    """
    metric_dict = {}
    for key in dataset_collection.keys():
        dataset = dataset_collection[key]
        feature = dataset.feature.numpy()
        target = dataset.target.numpy()
        
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(feature))
            
            if isinstance(pred, (tuple, list)):
                # extract the prediction for nn model
                if len(pred) != 2:
                    raise ValueError('neural model must return (features, forecast)')
                pred = pred[1]
            
            pred = pred.numpy()
        if pred.shape != target.shape:
            raise ValueError(f'forecast shape {pred.shape} does not match target shape {target.shape}')
        if not np.isfinite(pred).all() or not np.isfinite(target).all() or not len(target):
            raise ValueError('evaluation requires a nonempty batch of finite forecasts and targets')
            
        if dataset.is_scale:
            mean = dataset.target_mean.numpy()
            std = dataset.target_std.numpy()
            target = target * std + mean
            pred = pred * std + mean

        if loss == 'mape':
            if is_average:
                # print(np.max(np.abs(pred - target) / target))
                metric = np.mean(np.abs(pred - target) / target) * 100
            else:
                metric = np.mean(np.abs(pred - target) / target, axis=1) * 100
        elif loss == "mse":
            # todo: why the mse is unscaled here?
            if is_average:
                metric = np.mean((pred - target)**2)
            else:
                metric = np.mean((pred - target)**2, axis = 1)
        elif loss == 'cost':
            operator = Operator(case_config=case_config) # the operator is used to solve the power system operation problem
            pg, _, _, cost1, _ = operator.solve_one(pred)
            _, _, _, cost2, _ = operator.solve_two(target, pg)
            if is_average:
                metric = np.mean(cost1 + cost2)
            else:
                metric = cost1 + cost2
            
            if with_mispatch:
                gen_mismatch = pg.sum(axis = 1) - target.sum(axis = 1)
                metric_dict[key + '_gen_mismatch'] = gen_mismatch
            
        else:
            raise ValueError("loss should be either 'mape' or 'cost'")
        
        metric_dict[key] = metric
        
        if is_save:
            np.save(file = f"{save_dir}/{key}_{loss}.npy", arr = metric)

    return metric_dict

def flatten_model(model):
    """
    return the flattened parameters of a model
    """
    model.eval()
    flattened_params = []
    for param in model.parameters():
        flattened_params.append(param.flatten())
    
    return torch.cat(flattened_params).detach().numpy()

def reconstruct_model(model, flattened_params):
    model = copy.deepcopy(model)
    model.eval()
    idx = 0
    for param in model.parameters():
        param.data = torch.tensor(flattened_params[idx:idx+param.numel()]).reshape(param.shape).float()
        idx += param.numel()
    
    return model
