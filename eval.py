import torch
import torch.nn as nn






@torch.no_grad()
def test(model,mask,tensor_adjacency,tensor_x,tensor_y):
    model.eval()
    with torch.no_grad():
        out,_,_,_,_ = model(tensor_adjacency, tensor_x)
        loss = nn.CrossEntropyLoss()(out[mask], tensor_y[mask])
        pred = out.argmax(dim=1)
        correct = pred[mask].eq(tensor_y[mask]).sum().item()
        acc = correct / mask.sum().item()
    return acc,loss

