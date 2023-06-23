#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import random
from collections import deque

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CharLSTM
from utils import get_dataset, average_weights, exp_details
from shkp import ShakeSpeare



if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)
    
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    if args.dataset in ['mnist', 'cifar','fmnist']:
        train_dataset, test_dataset, user_groups = get_dataset(args)
    elif args.dataset == 'shakespeare':
        train_dataset = ShakeSpeare(train=True)
        test_dataset = ShakeSpeare(train=False)
        user_groups = train_dataset.get_client_dic()
        args.num_users = len(user_groups) # 139 users
        print(args.num_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')
    
    # For MNIST and CIFAR10, each edge has 5 clients
    # For Shakespeare, each edge has 14 clients and the last edge has 13
    num_edges = 10 
    step = 5 # clients/num_edges    

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,dim_out=args.num_classes)
    
    elif args.model == 'lstm':
        global_model = CharLSTM(args=args)
        step = 14 # 139 clients / 10 edge servers
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    
    # JJ inital queue for global model candidates
    global_models_q = deque()
    global_models_q.append(copy.deepcopy(global_model))
    
    # JJ inital models for all edge servers
    edge_models = [copy.deepcopy(global_model)]*num_edges
    
    for epoch in tqdm(range(args.epochs)):
        global_model.train()
        selected_edges_weights = []
        
        print(f'\n | Global Training Round : {epoch+1} |\n')
        
        # JJ select some edges to do training
        selected_edges = random.sample(range(num_edges),args.select_edges) # range creates sequence from 0-9 for edge servers
        print(selected_edges)
        # JJ initilized edge server models with global models
        global_models_list = list(global_models_q)
        for index, edge_index in enumerate(selected_edges):
            edge_models[edge_index] = copy.deepcopy(global_models_list[min(index,len(global_models_list)-1)])
        
        # JJ select clients to do training for each server and do aggregation
        for edge_index in selected_edges:
            # get clients for each selected edge server
            idxs_users = range(edge_index*step,min(len(user_groups),(edge_index+1)*step))
            # edge_weights, local_losses = [], [] # for each edge server, it has local weights and losses
            # get edge model for edge server
            edge_model = copy.deepcopy(edge_models[edge_index]) 
            edge_model.train().to(device)
            # FedAvg for all clients in edge server
            for ed_ep in range(args.edge_ep):
                local_weights = []
                for idx in idxs_users:
                    local_model = LocalUpdate(args=args, dataset=train_dataset,idxs=user_groups[idx], logger=logger)
                    w, loss = local_model.update_weights(model=copy.deepcopy(edge_model), global_round=epoch)
                    local_weights.append(copy.deepcopy(w))
                    #local_losses.append(copy.deepcopy(loss))
                edge_weights = average_weights(local_weights) # FedAvg weights
                edge_model.load_state_dict(edge_weights) # update edge server
            selected_edges_weights.append(copy.deepcopy(edge_weights))
            
            #loss_avg = sum(local_losses) / len(local_losses) # for one edge loss
            #train_loss.append(loss_avg) # for all edge loss
        
        # update global weights
        global_weights = average_weights(selected_edges_weights)
        # update global weights
        global_model.load_state_dict(global_weights)
        
        global_models_q.append(copy.deepcopy(global_model))
        # Maintain a dqueue with a size same as selected edges, and pop the oldest global model if needed
        if len(global_models_q) >= args.select_edges:
            global_models_q.popleft()
        
        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        
        #print("|---- Test Loss: {}".format(test_loss))
        print('| Global Round : {}|---- Test Accuracy: {:.2f}%'.format(epoch,100*test_acc))


       
        """
            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            global_model.eval()
            for c in range(args.num_users):
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=user_groups[idx], logger=logger)
                acc, loss = local_model.inference(model=global_model)
                list_acc.append(acc)
                list_loss.append(loss)
            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss))}')
                print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
        """
    
    """
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    """
    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
