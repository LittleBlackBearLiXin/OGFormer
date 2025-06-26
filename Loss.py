import torch
import torch.nn as nn
import torch.nn.functional as F




def kl_divergence_matrix(embeddings):
    embeddings = F.normalize(embeddings,dim=1, p=4)#p=1,but p=4 Good!

    log_p = torch.log(embeddings + 1e-10)
    p_log_p = torch.sum(embeddings * log_p, dim=1)
    cross_term = embeddings @ log_p.T
    kl_matrix = p_log_p.unsqueeze(+1) - cross_term

    return (kl_matrix)



from homous import Weight_homo_ratio

def neigh_loss(adj,logits,y,mask):
    updated_tensor_y = y.clone()
    updated_tensor_y[~mask] = logits.argmax(dim=1)[~mask]
    had1 = Weight_homo_ratio(F.normalize(adj, p=1, dim=1), updated_tensor_y)
    return torch.norm((1-had1))





class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self,embeddings,adj, lambda_reg):
        out=self.kl_Preg(embeddings, adj, lambda_reg)
        return out

    def kl_Preg(self,embeddings, adjacency_matrix, lambda_reg):
        diff = kl_divergence_matrix(embeddings)
        #diff=(diff+diff.T)*0.5#JS
        adjacency_matrix=F.normalize(adjacency_matrix, p=1)
        Preg = diff * adjacency_matrix
        Preg_loss = torch.sum(Preg)
        return lambda_reg * Preg_loss

