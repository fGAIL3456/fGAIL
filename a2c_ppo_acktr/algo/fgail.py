import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd

from baselines.common.running_mean_std import RunningMeanStd

class weightConstraint(object):
    '''weight clipping for W^(z)>0'''
    def __init__(self):
        pass
    def __call__(self,module):
        if hasattr(module, 'weight'):
            w=module.weight.data
            w=w.clamp(0)
            module.weight.data=w

class f_conjugate(nn.Module):
    def __init__(self, hidden_dim, lr, device):
        super(f_conjugate, self).__init__()
        self.device = device
        
        self.ylayer = nn.Linear(1, hidden_dim).to(device)
        self.zlayer1 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.zlayer2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.zlayer3 = nn.Linear(hidden_dim, 1).to(device)
        self.ylayer1 = nn.Linear(1, hidden_dim).to(device)
        self.ylayer2 = nn.Linear(1, hidden_dim).to(device)
        self.ylayer3 = nn.Linear(1, 1).to(device)
        self.bias = torch.tensor([0.], requires_grad=False)
        self.u = torch.tensor([0.], requires_grad=False)
        self.u_optimizer = torch.optim.Adam([self.u], lr=lr)
        self.shift()
    def forward(self, y):
        y = y - self.bias
        z1 = F.leaky_relu(self.ylayer(y), 0.1)
        z2 = F.leaky_relu(self.ylayer1(y) + self.zlayer1(z1), 0.1)
        z3 = F.leaky_relu(self.ylayer2(y) + self.zlayer2(z2), 0.1)
        z4 = F.leaky_relu(self.ylayer3(y) + self.zlayer3(z3), 0.1)
        return z4-self.bias
    def shift(self):
        for param in self.parameters():
            param.requires_grad = False
        self.u.requires_grad = True
        for _ in range(1):
            shift_loss = self.forward(self.u)-self.u
            self.u_optimizer.zero_grad()
            shift_loss.backward(retain_graph=True)
            self.u_optimizer.step()
        self.bias -= (self.forward(self.u)-self.u).detach()/2
        
class T(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(T, self).__init__()

        self.device = device
        
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)
        
    def forward(self, x):
        return self.trunk(x)
  
        
class Discriminator(nn.Module):
    '''
    expert: T(s,a)
    generated: f_conjugate(T(s,a))
    '''
    def __init__(self, input_dim, hidden_dim, lr, f_update, fnum, device):
        super(Discriminator, self).__init__()

        self.device = device
        self.f_conjugate = f_conjugate(hidden_dim, lr, device)
        self.T = T(input_dim, hidden_dim, device)
        self.f_conjugate.train()
        self.T.train()

        self.f_optimizer = torch.optim.Adam(self.f_conjugate.parameters(), lr=lr*0.001)
        self.T_optimizer = torch.optim.Adam(self.T.trunk.parameters())#, lr=lr)

        self.constraints = weightConstraint()
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self, expert_state, expert_action,
                         policy_state, policy_action, lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action.float()], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_d = self.f_conjugate(self.T(
                torch.cat([policy_state, policy_action.float()], dim=1))).mean()

            expert_state, expert_action = expert_batch
            expert_state = obsfilt(expert_state.numpy(), update=False)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.view((expert_state.shape[0], -1)).to(self.device)
            expert_d = self.T(
                torch.cat([expert_state, expert_action], dim=1)).mean()

            fgail_loss = - expert_d + policy_d
            loss += fgail_loss.item()

            n += 1

            self.f_optimizer.zero_grad()
            self.T_optimizer.zero_grad()
            fgail_loss.backward()
            self.f_optimizer.step()
            self.T_optimizer.step()

            self.f_conjugate._modules['zlayer1'].apply(self.constraints)
            self.f_conjugate._modules['zlayer2'].apply(self.constraints)
            self.f_conjugate._modules['zlayer3'].apply(self.constraints)
            self.f_conjugate.shift()

        return loss / n
    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            # should pass through f_conjugate(T)
            reward = self.f_conjugate(self.T(torch.cat([state, action.float()], dim=1)))
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
        