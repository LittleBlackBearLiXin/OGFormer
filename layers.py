import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_normal_(self.weight, gain=1)
        if self.use_bias:
            init.normal_(self.bias, mean=0.0, std=0.01)

    def forward(self,input_feature):
        output=torch.mm(input_feature,self.weight)
        if self.use_bias:
            output+=self.bias
        return output


class Comattion(nn.Module):
    def __init__(self, input_dim,output,alpha):
        super(Comattion, self).__init__()
        self.linq=MLP(input_dim,output)
        self.alpha=alpha
        self.reset_parameters()
    def reset_parameters(self):
        self.linq.reset_parameters()

    def forward(self, adjacency,feature):

        hq = F.sigmoid(self.linq(feature))
        R = self.compute_att(hq,adjacency,self.alpha)

        return hq, R


    def compute_att(self, X,SE,alpha):
        X = (X - X.mean(dim=1, keepdim=True)) / (X.std(dim=1, keepdim=True) + 1e-8)
        intersection = torch.mm(X, X.T)**2
        att = F.normalize(intersection, p=1, dim=1)
        R=(att+alpha*SE)

        return R


class MessgePsssing(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True,SYSnoram=False):
        super(MessgePsssing, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.SYSnoram=SYSnoram
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight, gain=1)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        if self.SYSnoram:
            adjacency = self.normalize_adjacency(adjacency)
        else:
            adjacency=F.normalize(adjacency,p=1)
        support = torch.mm(adjacency, input_feature)
        output = torch.mm(support, self.weight)
        if self.use_bias:
            output += self.bias
        return output
    def normalize_adjacency(self,adjacency):
        d_inv_sqrt = torch.pow(torch.sum(adjacency, dim=1), -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adjacency_normalized = torch.mm(torch.mm(d_mat_inv_sqrt, adjacency), d_mat_inv_sqrt)

        return adjacency_normalized










