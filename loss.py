#loss
import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_function_mse(generated, real):
  mse_loss = nn.MSELoss()
  loss = mse_loss(generated, real)
  return loss

''' 
def weighted_mse_loss(input, target, weights):
  
    out = (input - target)**2
    #print(out.shape) = torch.Size([20, 3, 800])
    out = out * weights.expand_as(out) #same dimensionality
    loss = out.mean() 
    return loss

def loss_function_mse(generated, real, start_weight=5, end_weight=1):
    #print(generated.shape)
    sequence_length = generated.size(2)
    penalty_length = int(sequence_length * 0.25)
    penalty = 5
    normal_weight = 1
    #print(sequence_length)
    #weights = torch.linspace(start_weight, end_weight, steps=sequence_length)
    first_weights = torch.full((penalty_length,), penalty)
    rest_weights =  torch.full((sequence_length - penalty_length,), normal_weight)
    
    weights = torch.cat((first_weights, rest_weights), dim=0)
    
    if generated.is_cuda:
        weights = weights.to(generated.device)
    
    mse_loss = weighted_mse_loss(generated, real, weights)
    return mse_loss
'''     

def loss_function_adv(output, target):
  loss = F.binary_cross_entropy_with_logits(output, target) # or nn.BCEWithLogitsLoss()
  return loss

def loss_L1(input,target):
    loss = nn.L1Loss()
    output = loss(input,target)
    return output
