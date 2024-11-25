'''loss functions'''
import torch

# -------------------------------------------------------
# loss function for M
def l2norm(Mtarget,Msim,weight='none'):
    '''
    Args:
        Mtarget: (3,N)
        Msim: (3,N)
        weight: (1,N) or none
    '''
    err = torch.linalg.norm((Mtarget - Msim),dim=0)**2
    if weight == 'none':
        N = Mtarget.shape[1]
        loss = torch.sum(err)/N
    else:
        N = torch.count_nonzero(weight)
        # M = torch.mean(weight)
        loss = torch.sum(err*weight)/(N)
    return loss

def vector_disalignment(Mtarget,Msim,weight='none'):
    '''
    input:
        Mtarget: (3*N)
        Msim: (3*N)
        weight: (1*N) or none
    '''
    # use 1 minus the inner product, thus give the objective always positive
    # and we want to minimize the objective
    innerp = 1. - torch.sum(Mtarget*Msim,dim=0)
    if weight == 'none':
        N = Mtarget.shape[1]
        loss = torch.sum(innerp)/N
    else:
        N = torch.count_nonzero(weight)
        # M = torch.sum(weight)/N
        loss = torch.sum(innerp*weight)/(N)
    return loss

# -------------------------------------------------------


# -------------------------------------------------------
# loss function for spin-domain parameters
def loss_3term(tar_para_1,tar_para_2,tar_para_3,para_1,para_2,para_3,weight):
    '''loss function for three terms parameter'''
    loss = (tar_para_1-para_1)**2 + (tar_para_2-para_2)**2 + (tar_para_3-para_3)**2
    loss = torch.sum(loss*weight)
    return loss

def loss_2term(tar_para_1,tar_para_2,para_1,para_2,weight):
    '''loss function for two terms parameter'''
    loss = (tar_para_1-para_1)**2 + (tar_para_2-para_2)**2
    loss = torch.sum(loss*weight)
    return loss


