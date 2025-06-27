import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

zero=torch.tensor(0.0).to(DEVICE)
one=torch.tensor(1.0).to(DEVICE)
alpha=torch.tensor(1e-8).to(DEVICE)
def Neigh_homo_ratio(adjaceny, Y):
    adjaceny=adjaceny.to(DEVICE)
    Y=Y.to(DEVICE)

    A=torch.where(adjaceny!=0,one, zero)
    Y=Y+1
    Y = Y.float()

    A_Y = torch.diag(Y)



    L_matrix =Y.unsqueeze(1).expand_as(A_Y)
    B = torch.mm(A, A_Y)

    C=torch.where(B==L_matrix,adjaceny,zero)
    D = torch.sum(C, dim=1)

    A_num = torch.sum(adjaceny, dim=1)

    H = torch.where(A_num == 0, zero, D / A_num)
    l = len(Y)

    nero=torch.sum(A_num==0)
    l=l-nero
    return torch.sum(H)/l


def Weight_homo_ratio(adjaceny, Y):
    adjaceny=adjaceny.to(DEVICE)
    Y=Y.to(DEVICE)

    A=torch.where(adjaceny!=0,one, zero)
    Y=Y+1
    Y = Y.float()

    A_Y = torch.diag(Y)



    L_matrix =Y.unsqueeze(1).expand_as(A_Y)
    B = torch.mm(A, A_Y)

    C=torch.where(B==L_matrix,adjaceny,zero)
    D = torch.sum(C, dim=1)

    A_num = torch.sum(adjaceny, dim=1)

    H = torch.where(A_num == 0, zero, D / A_num)
    H = torch.where(H == 0, alpha, H)


    return H


def homophily_add(adjacency,labels):

    adjacency1 = torch.where(adjacency > zero, one, zero)
    labels=labels+one
    Y = torch.diag(labels).float().to(DEVICE)

    A_Y = torch.mm(adjacency1,Y)

    Y_Y = labels.unsqueeze(1).expand_as(A_Y)


    filtered_A_Y = torch.where(A_Y == Y_Y, adjacency,zero)
    filtered_A_Y=torch.where(filtered_A_Y > zero, one, zero)


    return filtered_A_Y
