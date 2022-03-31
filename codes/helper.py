from uncertainties_tools import *
from sgld import *
import sgld_tools
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.functional as F
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torchvision import datasets
from torchvision.transforms import ToTensor

from sklearn.decomposition import TruncatedSVD, PCA
from scipy.stats import multivariate_normal
import copy

import matplotlib.pyplot as plt

import os

import models
from swag import *

def Predictive_entropy(ytest, all_probs, post_net, loader_test, path_fig = None):
    # Compute the Negative Log Likelihood (NLL)
    # ytest = testloader.dataset.targets
    log_it = - np.log(np.take_along_axis(all_probs, np.expand_dims(ytest, axis=1), axis=1)).squeeze()
    nll = log_it.mean()
    entropy_dataset = - (all_probs * np.log(all_probs)).sum(axis=1)

    # Compute the predicted probabilities
    predicted_ood = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for images, labels in tqdm(loader_test):
            images, labels = images.to(device), labels.to(device)
            outputs = post_net(images).cpu().numpy()
            if predicted_ood is None:
                predicted_ood = outputs
            else:
                predicted_ood = np.vstack((predicted_ood, outputs))

    # Compute the Negative Log Likelihood (NLL) on MNIST
    # entropy_ood = - np.log(np.take_along_axis(predicted_ood, np.expand_dims(ood_test, axis=1), axis=1)).squeeze()
    entropy_ood = - (predicted_ood * np.log(predicted_ood)).sum(axis=1)

    # Store the results in a dictionary    
    entropy_dict = {"Dataset": entropy_dataset, "OOD dataset": entropy_ood}

    # Display the predicted entropies
    if path_fig is not None:
        ax = sns.kdeplot(data=entropy_dict, fill=True, cut=0, common_norm=False)
        ax.set(xlabel='pred. entropy', ylabel='Density')  # title='')
        plt.savefig(path_fig, bbox_inches='tight')
        plt.show()

    return entropy_dict, nll


def train_epoch(train_loader, test_loader, model, criterion, optimizer, title):
    train_res = {'loss' : 0, 'accuracy' : 0}
    model.train()
    for idx, (train_x, train_label) in enumerate(train_loader):
        label_np = np.zeros((train_label.shape[0], 10))
        optimizer.zero_grad()
        predict_y = model(train_x.float())
        loss = criterion(predict_y, train_label.long())
        if idx % 10 == 0:
            print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
        loss.backward()
        optimizer.step()

    correct = 0
    _sum = 0
    model.eval()
    for idx, (test_x, test_label) in enumerate(test_loader):
        predict_y = model(test_x.float()).detach()
        predict_ys = np.argmax(predict_y, axis=-1)
        label_np = test_label.numpy()
        _ = predict_ys == test_label
        correct += np.sum(_.numpy(), axis=-1)
        _sum += _.shape[0]

    train_res['loss'] = loss.item()
    train_res['accuracy'] = correct / _sum
    torch.save(model.state_dict(), "ckpts/" + title + ".pt")
    return train_res


def train(model, train_loader, test_loader, optimizer, criterion, lr_init=1e-2, epochs=3000, title="",
          swag_model=None, swag=False, swag_start=2000, swag_freq=50, swag_lr=1e-3,
          print_freq=10):
    
    for epoch in range(epochs):
        t = (epoch + 1) / swag_start if swag else (epoch + 1) / epochs
        lr_ratio = swag_lr / lr_init if swag else 0.05
        
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio

        lr = factor * lr_init
        adjust_learning_rate(optimizer, lr)
        
        train_res = train_epoch(train_loader, test_loader, model, criterion, optimizer, title)
        if swag and epoch > swag_start:
            swag_model.collect_model(model)
        
        if (epoch % print_freq == 0 or epoch == epochs - 1):
            print('Epoch %d. LR: %g. Loss: %.4f Accuracy %.4f' % (epoch, lr, train_res['loss'], train_res['accuracy']))
        
        
def model_averaging(swag_model, model, loader, S=100):
    x_ = loader.dataset.data
    all_probs = np.zeros((x_.shape[0], 10))
    for i in range(S):
        weights = swag_model.sample()
        set_weights(model, weights)
        res = []
        for train_x, y in loader:
            res.append(nn.Softmax(dim=1)(model(train_x)).detach().numpy())
        all_probs += np.vstack(res)
    all_probs /= S
    return all_probs


def diag_model_averaging(swag_model, model, loader, S=100):
    x_ = loader.dataset.data
    all_probs = np.zeros((x_.shape[0], 10))
    for i in range(S):
        weights = swag_model.diag_sample()
        set_weights(model, weights)
        res = []
        for test_x, y in loader:
            res.append(nn.Softmax(dim=1)(model(test_x)).detach().numpy())
        all_probs += np.vstack(res)
    all_probs /= S
    return all_probs


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy_all_probs(all_probs, ytest):
    correct = 0
    total = 0
    for y_pred, labels in zip(all_probs, ytest):
        labels = labels.reshape(-1, 1)
        predicted = np.argmax(y_pred, axis=-1)
        total += labels.shape[0]
        correct += np.sum(predicted == labels, axis=-1)
    return 100 * correct / total



def noniid_datasets(train_loader, n_clients=10, n_classes=10, proportion=0.25, batch_size=64):
    n_sizes = np.sort(np.random.uniform(size=n_clients)).reshape(-1, 1) # Assume the size is increasing (because why not)
    interests = proportion * np.ones((n_clients, n_classes))
    for i in range(n_classes):
        interests[i, i % n_classes] = 1
    statistics = interests * n_sizes
    statistics = statistics/statistics.sum(axis=0, keepdims=1)
    resulting_dataset = {i: [[], []] for i in range(n_clients)}
    for X, Y in train_loader:
        for x, y in zip(X, Y):
            client = np.random.choice([i for i in range(n_clients)], 1, p=[statistics[i, y.item()] for i in range(n_clients)])[0]
            resulting_dataset[client][0].append(x[None])
            resulting_dataset[client][1].append(y.reshape(1, -1))
    for i in range(n_clients):
        if resulting_dataset[i][0]:
            resulting_dataset[i][0] = torch.cat(resulting_dataset[i][0])
            resulting_dataset[i][1] = torch.cat(resulting_dataset[i][1]).view(-1)
            train_dataset_i = TensorDataset(resulting_dataset[i][0], resulting_dataset[i][1])
            resulting_dataset[i] = DataLoader(train_dataset_i, batch_size=batch_size)
    return resulting_dataset


def visualizing_client_loader(client_loaders, n_clients=10, n_classes=10, path_figures=None):
    fig, axs = plt.subplots(2, n_clients//2, constrained_layout=True, figsize=(20, 8))
    fig.suptitle("Clients' local datasets heterogeneity", fontsize=20)
    axs[0, 0].set_ylabel("Count", fontsize=16)
    axs[1, 0].set_ylabel("Count", fontsize=16)
    
    sizes = [0 for _ in range(n_clients)]
    
    for i in range(n_clients):
        client_i = client_loaders[i]
        counts = [0 for _ in range(n_classes)]
        for X, Y in client_i:
            for x, y in zip(X, Y):
                counts[y.item()] += 1
        sizes[i] = sum(counts)
        k, j = i % (n_clients // 2), i // (n_clients // 2)
        axs[j, k].get_xaxis().set_ticks(range(n_classes + 1))
        axs[j, k].get_yaxis().set_ticks([])
        axs[j, k].set_xlabel("classes", fontsize=16)
        axs[j, k].bar(range(n_classes), counts)
    if path_figures:
        plt.savefig(path_figures + 'heterogeneity_clients_local_datasets.pdf', bbox_inches='tight')
    plt.show()
    
    fig = plt.figure(figsize=(4, 2))
    plt.title("Sizes of local datasets")
    plt.xlabel("clients")
    plt.ylabel("local dataset size")
    plt.xticks(range(n_clients + 1))
    plt.yticks(fontsize=8)
    plt.bar(range(n_clients), sizes)
    if path_figures:
        plt.savefig(path_figures + 'imbalanced_local_datasets_sizes.pdf', bbox_inches='tight')
    plt.show()

def save_calibration_scores(all_probs, ytest, path_variables="./variables", title=""):
    # Compute the final accuracy of the averaged model from all_probs(bayesian learning)
    final_acc = accuracy_all_probs(all_probs, ytest)

    # Compute the accuracy in function of p(y|x)>tau
    tau_list = np.linspace(0, 1, num=100)
    accuracies, misclassified = confidence(ytest, all_probs, tau_list)

    # Compute the Expected Calibration Error (ECE)
    ece = ECE(all_probs, ytest, num_bins = 20)

    # Compute the Brier Score
    bs = BS(ytest, all_probs)

    # Compute the accuracy - confidence
    acc_conf = accuracy_confidence(all_probs, ytest, tau_list, num_bins = 20)

    # Compute the calibration curve
    cal_curve = calibration_curve(all_probs, ytest, num_bins = 20)
    
    # Compute negative loglikelihood
    log_it = - np.log(np.take_along_axis(all_probs, np.expand_dims(ytest, axis=1), axis=1)).squeeze()
    nll = log_it.mean()
    
    file = open(path_variables + "/text" + title + '.txt', 'a')
    file.write(f"\nFinal accuracy = {final_acc}, \nECE = {ece}, \nBS = {bs}, \nNLL = {nll}")
    file.close()  # to change the file access mode

    
def save_accuracies(model_cfg, Mu_s, accuracies, test_loader, variables_path=None):
    for i, mu in enumerate(Mu_s):
        model = model_cfg.base(*model_cfg.args, **model_cfg.kwargs)
        set_weights(model, torch.tensor(mu))
        accuracies[f"client {i}"] = accuracy_model(model, test_loader, 'cpu')
    
    file = open(variables_path +  'distributedLogisticRegression' + '.txt', 'a')
    file.write("--------------- New Run ---------------\n")
    for key, value in accuracies.items():
        file.write(f"{key}: {value}\n")
    file.close()  # to change the file access mode

    
    
    
import time

import numpy as np
import torch
from sklearn.utils import shuffle
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T

def heterogeneous_dataset(trainloader, num_clients = 20, proportion = .25):
    print("actually newer new")
    start_time = time.time()

    inputs = None
    targets = None
            
    for data in trainloader:
        if inputs is None:
            inputs, targets = data[0], data[1]

        else:
            inputs, targets = torch.cat((inputs, data[0])), torch.cat((targets, data[1]))

    intermediate_time = time.time()
    # print(intermediate_time - start_time)

    # Generate the imbalanced datasets
    datasets_dict = {}
    classes = np.unique(targets)
    num_classes = len(classes)
    for c, label in enumerate(classes):
        index_label = np.where(targets == label)[0]
        num_data = len(index_label)
        # number of data associated with this label on the clients for which the main class corresponds to this label
        num_main = int(proportion * num_data)
        num_label = num_clients // num_classes + (c < (num_clients % num_classes))
        num_minor = (num_data - num_main * num_label) // (num_clients - num_label)
        split_pt = 1
        split_list = [0]
        reminder = num_data - num_main * num_label - num_minor * (num_clients - num_label)
        for w in range(num_clients):
            split_pt += num_main if (w % num_classes == c) else num_minor
            if reminder > 0:
                split_pt += 1
                reminder -= 1
            split_list.append(split_pt - 1)
        # split_list[-1] += 1
        for w in range(num_clients):
            ind_w = index_label[split_list[w]: split_list[w + 1]]
            if c == 0:
                datasets_dict[w] = [inputs[ind_w], targets[ind_w]]
            else:
                datasets_dict[w][0] = torch.cat((datasets_dict[w][0], inputs[ind_w]))
                datasets_dict[w][1] = torch.cat((datasets_dict[w][1], targets[ind_w]))

    for key, item in datasets_dict.items():
        print(key, item[1])
        item[0], item[1] = shuffle(item[0], item[1])

    datasets = []
    for x, y in datasets_dict.values():
        datasets.append([x, y])

    for i, (x, y) in enumerate(datasets):
        print(f"Worker number {i}, {len(y)}, \n {y}\n")
    print([[len(np.where(datasets[w][1] == c)[0]) for c in classes] for w in range(num_clients)])
    # print('Total time to generate the imbalanced dataset:', time.time() - intermediate_time)
    
    
    client_train_loaders_vincent = [0 for _ in range(num_clients)]
    for i in range(num_clients):
        dataset_vincent = TensorDataset(datasets[i][0], datasets[i][1])
        client_train_loaders_vincent[i] = DataLoader(dataset_vincent, batch_size=64)
    return client_train_loaders_vincent




import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import HMC, MCMC, NUTS









