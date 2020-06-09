import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BC():
    def __init__(self, actor_critic, num_mini_batch, lr=None, eps=None,
                 max_grad_norm=None, use_clipped_value_loss=True, device=None):

        self.actor_critic = actor_critic
        self.num_mini_batch = num_mini_batch
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, train_loader, obsfilt=None):
        action_loss_epoch = 0

        for obs_batch, actions_batch in train_loader:
            # Reshape to do in a single forward pass for all steps
            obs_batch = obsfilt(obs_batch.numpy(), update=False)
            obs_batch = torch.FloatTensor(obs_batch).to(self.device)
            actions_batch = actions_batch.view((actions_batch.shape[0], -1)).to(self.device)
            
            values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                obs_batch, None, None, actions_batch)

            action_loss = -action_log_probs.mean()

            self.optimizer.zero_grad()
            action_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)
            self.optimizer.step()

            action_loss_epoch += action_loss.item()
            
        action_loss_epoch /= self.num_mini_batch

        return _, action_loss_epoch, _
