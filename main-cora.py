import torch
import numpy as np
from load_data import load_data
import torch.nn as nn
import torch.nn.functional as F
from layers import MessgePsssing,Comattion
from Nodepre import preprocess

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_data(name='Cora',use_planetoid=True,use_geom=False,use_webkb=False,use_wikipedia_network=False)

data = dataset[0]
print(data)
indim=data.num_nodes

data = preprocess(data,loop=False)

print(len(data.train_mask[data.train_mask == True]))
print(len(data.val_mask[data.val_mask == True]))
print(len(data.test_mask[data.test_mask == True]))
print('============================================================================')





class OGFormer(nn.Module):
    def __init__(self, input_dim):
        super(OGFormer, self).__init__()
        self.comatt1= Comattion(input_dim,64,0.8)
        self.comatt2 = Comattion(64, 64,0.9)
        self.agg1 = MessgePsssing(input_dim, 64,SYSnoram=False)
        self.agg2 = MessgePsssing(64, 64,SYSnoram=False)
        self.reset_parameters()
    def reset_parameters(self):
        print('model reset parameters')
        self.comatt1.reset_parameters()
        self.comatt2.reset_parameters()
        self.agg1.reset_parameters()
        self.agg2.reset_parameters()

    def forward(self, adjacency,feature):
        hq1,R1 = self.comatt1(adjacency,feature)
        h1 = F.relu(self.agg1(R1, feature))+hq1
        hq2, R2 = self.comatt2(R1, h1)
        h2 = (self.agg2(R2, h1))+hq2

        return F.log_softmax(h2, dim=1), hq1, R1, hq2, R2





import torch.optim as optim

LEARNING_RATE = 0.01
WEIGHT_DACAY = 8e-4
EPOCHS = 200
#Epoch 199 Average Test Accuracy: 0.8641,Average Test std: 0.0038


tensor_x = data.x.to(DEVICE)
tensor_y = data.y.to(DEVICE)
tensor_adjacency=data.adjacency.to(DEVICE)
tensor_train_mask = data.train_mask.to(DEVICE)
tensor_val_mask = data.val_mask.to(DEVICE)
tensor_test_mask = data.test_mask.to(DEVICE)



model = OGFormer(data.x.shape[1]).to(DEVICE)






from eval import test

from torch.optim.lr_scheduler import ReduceLROnPlateau
from Loss import Loss,neigh_loss

import torch.backends.cudnn as cudnn
import random
def train(iteration_seed):
    torch.manual_seed(iteration_seed)
    np.random.seed(iteration_seed)
    random.seed(iteration_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    epoch_test_acc_history = [[] for _ in range(EPOCHS)]

    model.reset_parameters()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DACAY)
    torch.cuda.empty_cache()
    cross_loss= nn.CrossEntropyLoss()
    klloss = Loss()

    scheduler = ReduceLROnPlateau(optimizer, min_lr=1e-4, factor=0.6, patience=2, verbose=True)


    for epoch in range(EPOCHS):
        model.train()
        logits,h11,adj1,h22,adj2 = model(tensor_adjacency, tensor_x)

        train_mask_logits = logits[tensor_train_mask]

        negiloss1=neigh_loss(adj1, logits, tensor_y, tensor_train_mask)
        negiloss2=neigh_loss(adj2,logits,tensor_y,tensor_train_mask)

        loss = cross_loss(train_mask_logits, tensor_y[tensor_train_mask])+klloss(h11,adj1,1e-4)+negiloss1*1e-4\
               +klloss(h22,adj2,1e-4)+negiloss2*1e-4


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc,_ = test(model,tensor_train_mask,tensor_adjacency,tensor_x,tensor_y)
        val_acc,val_loss = test(model,tensor_val_mask,tensor_adjacency,tensor_x,tensor_y)
        test_acc,_ = test(model,tensor_test_mask,tensor_adjacency,tensor_x,tensor_y)
        scheduler.step(val_loss)


        epoch_test_acc_history[epoch].append(test_acc)
        print(f"Epoch {epoch}: Loss {loss.item():.4f},  Train Acc {train_acc:.4f},Val Loss {val_loss.item():.4f}, Val Acc {val_acc:.4f}, Test Acc {test_acc:.4f}")
    torch.cuda.empty_cache()
    return epoch_test_acc_history






def print_final_results():
    test_acc_list = []
    max_test_acc_list = []
    all_epochs_test_acc_history = [[] for _ in range(EPOCHS)]
    for iteration in range(0, 100):
        print(f"Now it's the {iteration}th iteration")


        epoch_test_acc_history = train(iteration)
        for epoch in range(EPOCHS):
            all_epochs_test_acc_history[epoch].extend(epoch_test_acc_history[epoch])
            test_acc = epoch_test_acc_history[-1][-1]
            test_acc_list.append(test_acc)
            max_test_acc = max(max(epoch_acc) for epoch_acc in epoch_test_acc_history)
            max_test_acc_list.append(max_test_acc)

        test_acc_avg = np.mean(test_acc_list)
        test_acc_std = np.std(test_acc_list)
        test_acc_max = np.max(test_acc_list)
        test_acc_min = np.min(test_acc_list)
        max_test_acc_avg = np.mean(max_test_acc_list)
        max_test_acc_std = np.std(max_test_acc_list)
        max_test_acc_max = np.max(max_test_acc_list)
        max_test_acc_min = np.min(max_test_acc_list)
    print("Average Test Accuracy over 100 training runs: {:.4f}".format(test_acc_avg))
    print("Standard Deviation of Test Accuracy over 100 training runs: {:.4f}".format(test_acc_std))
    print('Max Test Accuracy:', test_acc_max)
    print('Min Test Accuracy:', test_acc_min)
    print('==================================================================')
    print("Average Max Test Accuracy over 100 training runs: {:.4f}".format(max_test_acc_avg))
    print("Standard Deviation of Max Test Accuracy over 100 training runs: {:.4f}".format(max_test_acc_std))
    print('Max of Max Test Accuracy:', max_test_acc_max)
    print('Min of Max Test Accuracy:', max_test_acc_min)


    for epoch in range(EPOCHS):
        avg_test_acc = np.mean(all_epochs_test_acc_history[epoch])
        avg_test_std = np.std(all_epochs_test_acc_history[epoch])
        print(f"Epoch {epoch} Average Test Accuracy: {avg_test_acc:.4f},Average Test std: {avg_test_std:.4f}")
print_final_results()