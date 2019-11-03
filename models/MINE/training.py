import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.autograd as autograd

def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et

def learn_mine(batch, mine_net, mine_net_optim,  ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    joint , marginal = batch
    joint = torch.FloatTensor(joint).cuda()
    marginal = torch.FloatTensor(marginal).cuda()
    mi_lb , t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1 - ma_rate)*ma_et + ma_rate*torch.mean(et)

    # unbiasing use moving average
    loss = - (torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))
    # use biased estimator
    # loss = - mi_lb

    mine_net_optim.zero_grad()
    autograd.backward(loss)
    mine_net_optim.step()
    return mi_lb, ma_et

def sample_batch(data, batch_size=100, sample_mode='joint'):
    data_idx = range(data.shape[0])
    if sample_mode == 'joint':
        index = np.random.choice(data_idx, size=batch_size, replace=False)
        batch = data[index]
    else:
        joint_index = np.random.choice(data_idx, size=batch_size, replace=False)
        marginal_index = np.random.choice(data_idx, size=batch_size, replace=False)
        batch = np.concatenate([data[joint_index][:,0].reshape(-1,1),
                                data[marginal_index][:,1].reshape(-1,1)], axis=1)
    return batch

def train(data, mine_net,mine_net_optim, batch_size=100, iter_num=int(5e+3), log_freq=int(1e+3)):
    # data is x or y
    result = list()
    ma_et = 1.
    for i in range(iter_num):
        batch = sample_batch(data,batch_size=batch_size)\
        , sample_batch(data,batch_size=batch_size,sample_mode='marginal')
        mi_lb, ma_et = learn_mine(batch, mine_net, mine_net_optim, ma_et)
        result.append(mi_lb.detach().cpu().numpy())
        if (i+1)%(log_freq)==0:
            print(result[-1])
    return result

def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]