# individual network settings for each actor + critic pair
# see networkforall for details
'''
An addaption from:

Code partially extracted from:
https://github.com/denisyarats/pytorch_sac/blob/81c5b536d3a1c5616b2531e446450df412a064fb/agent/sac.py
https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/SAC/sac_torch.py
https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py


'''

from  rl_algorithms.networkforall_sac_qmix import Network, Network_discretaction
from  rl_algorithms.utilities import hard_update
# from utilities import gumbel_softmax, onehot_from_logits
from torch.optim import Adam, AdamW
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


# add OU noise for exploration
from  rl_algorithms.OUNoise import OUNoise

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

DISCRETE_ACTIONS = False

class SACQMIXAgent():
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, rnn_num_layers, rnn_hidden_size_actor, rnn_hidden_size_critic , lr_actor=1.0e-2, lr_critic=1.0e-2, weight_decay=1.0e-5, device = 'cpu', rnn = True, alpha = 0.2, automatic_entropy_tuning = True):
        super(SACQMIXAgent, self).__init__()

        if DISCRETE_ACTIONS == False:
            self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, rnn_num_layers, rnn_hidden_size_actor, device,actor=True, rnn = rnn).to(device)
            self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1, rnn_num_layers, rnn_hidden_size_critic, device, rnn = rnn).to(device)
            # self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, rnn_num_layers, rnn_hidden_size_actor, device, actor=True, rnn = rnn).to(device)
            self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1, rnn_num_layers, rnn_hidden_size_critic, device, rnn = rnn).to(device)
        else:
            self.actor = Network_discretaction(in_actor, hidden_in_actor, hidden_out_actor, out_actor, rnn_num_layers, rnn_hidden_size_actor, device,actor=True, rnn = rnn).to(device)
            self.critic = Network_discretaction(in_critic, hidden_in_critic, hidden_out_critic, 1, rnn_num_layers, rnn_hidden_size_critic, device, rnn = rnn).to(device)
            # self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, rnn_num_layers, rnn_hidden_size_actor, device, actor=True, rnn = rnn).to(device)
            self.target_critic = Network_discretaction(in_critic, hidden_in_critic, hidden_out_critic, 1, rnn_num_layers, rnn_hidden_size_critic, device, rnn = rnn).to(device)

        self.noise = OUNoise(out_actor, scale=1.0 )
        self.device = device
        
        # from torchsummary import summary
        
        # import pdb; pdb.set_trace()
        # summary(self.actor, (3, 224, 224))

        # initialize targets same as original networks
        # hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)
        # self.actor_optimizer = AdamW(self.actor.parameters(), lr=lr_actor, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        # self.critic_optimizer = AdamW(self.critic.parameters(), lr=lr_critic, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        
        # Alpha 
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.alpha = alpha
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(out_actor).to(self.device)).item()
            self.log_alpha = (torch.zeros(1, requires_grad=True, device=self.device)+np.log(self.alpha)).detach().requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=lr_actor)
            
            

    def act(self, his, obs, noise=0.0):
        his = his.to(self.device)
        obs = obs.to(self.device)
        # import pdb; pdb.set_trace()
        if DISCRETE_ACTIONS == False:
            if noise > 0.0:
                action, _ = self.actor.sample_normal(his,obs) 
            else:
                action, _ = self.actor.forward(his,obs) 
                action = action.cpu().clamp(-1, 1)
        else:
            actions = self.actor.forward(his,obs)
            if noise > 0.0:
                # log_probs, actions = self.calc_log_prob_action(self, actions, reparam=True)
                action = self.gumbel_softmax(actions,hard=True).argmax(dim=-1).reshape(obs.shape[0],1).float().detach() -1. #the -1 at the end is to have an action equal to -1,0,1 
                # log_probs = self.log_prob(actions,action)
            else:
                # log_probs, actions = self.calc_log_prob_action(self, actions, reparam=False)
                action = self.gumbel_softmax(actions,hard=False).argmax(dim=-1).reshape(obs.shape[0],1).float().detach() -1. #the -1 at the end is to have an action equal to -1,0,1 
                # log_probs = self.log_prob(actions,action)
        return action.cpu()
    

    def act_prob(self, his, obs, noise=0.0):
        his = his.to(self.device)
        obs = obs.to(self.device)
        # import pdb; pdb.set_trace()
        if DISCRETE_ACTIONS == False:
            #before 5/12/2022
            # actions, log_probs = self.actor.sample_normal(his,obs) 
            #After 5/12/2022
            #from https://github.com/kengz/SLM-Lab/blob/dda02d00031553aeda4c49c5baa7d0706c53996b/slm_lab/agent/algorithm/sac.py
            #and https://medium.com/@kengz/soft-actor-critic-for-continuous-and-discrete-actions-eeff6f651954
            if noise > 0.0:
                action, log_probs = self.actor.sample_normal(his,obs) 
            else:
                action, log_probs = self.actor.forward(his,obs) 
                action = action.cpu().clamp(-1, 1)
        else:
            actions = self.actor.forward(his,obs)
            if noise > 0.0:
                # log_probs, actions = self.calc_log_prob_action(self, actions, reparam=True)
                action = self.gumbel_softmax(actions,hard=True).argmax(dim=-1)
                log_probs = self.log_prob(actions,action)
                action = action.reshape(obs.shape[0],1).float().detach() - 1. #the -1 at the end is to have an action equal to -1,0,1 
            else:
                # log_probs, actions = self.calc_log_prob_action(self, actions, reparam=False)
                action = self.gumbel_softmax(actions,hard=False).argmax(dim=-1)
                log_probs = self.log_prob(actions,action)
                action = action.reshape(obs.shape[0],1).float().detach() - 1. #the -1 at the end is to have an action equal to -1,0,1 
        # import pdb; pdb.set_trace()
        return action.cpu(), log_probs 
    
    # def calc_log_prob_action(self, action_pd, reparam=False):
    #     '''Calculate log_probs and actions with option to reparametrize from paper eq. 11'''
    #     actions = self.rsample(action_pd) if reparam else self.sample(action_pd)
    #     log_probs = self.log_prob(actions)
    #     return log_probs, actions
    
    # def sample(self, logits, sample_shape=torch.Size()):
    #     '''Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical'''
    #     u = torch.empty(logits.size(), device=logits.device, dtype=logits.dtype).uniform_(0, 1)
    #     noisy_logits = logits - torch.log(-torch.log(u))
    #     return torch.argmax(noisy_logits, dim=-1)
    
    # def rsample(self, logits, sample_shape=torch.Size()):
    #     '''
    #     Gumbel-softmax resampling using the Straight-Through trick.
    #     Credit to Ian Temple for bringing this to our attention. To see standalone code of how this works, refer to https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
    #     '''
    #     rout = distributions.RelaxedOneHotCategorical.rsample(sample_shape)  # differentiable
    #     out = F.one_hot(torch.argmax(rout, dim=-1), logits.shape[-1]).float()
    #     return (out - rout).detach() + rout

    def log_prob(self, logits, value):
        '''value is one-hot or relaxed'''
        if value.shape != logits.shape:
            value = F.one_hot(value.long(), logits.shape[-1]).float()
            assert value.shape == logits.shape
        return - torch.sum(- value * F.log_softmax(logits, -1), -1)

    def onehot_from_logits(self,logits, eps=0.0):
        """
        Given batch of logits, return one-hot sample using epsilon greedy strategy
        (based on given epsilon)
        """
        # get best (according to current policy) actions in one-hot form
        argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
        return argmax_acs
    
    def sample_gumbel(self,shape, eps=1e-20, tens_type=torch.FloatTensor):
        """Sample from Gumbel(0, 1)"""
        U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
        return -torch.log(-torch.log(U + eps) + eps)
    
    # modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    def gumbel_softmax_sample(self,logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self.sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
        return F.softmax(y / temperature, dim=-1)
    
    # modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    def gumbel_softmax(self,logits, temperature=1.0, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
    
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            y_hard = self.onehot_from_logits(y)
            y = (y_hard - y).detach() + y
        return y
