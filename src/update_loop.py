import torch
import numpy as np
from typing import List, Union
from .agent import ValueNetwork, SoftQvalueNetwork, PolicyNetwork

def update(state: np.ndarray,  action: np.ndarray,  reward: np.ndarray,  next_state: np.ndarray,  done: np.ndarray, 
           value_network: ValueNetwork,  target_value_network: ValueNetwork, 
           soft_q_network1: SoftQvalueNetwork, soft_q_network2: SoftQvalueNetwork, 
           policy_network: PolicyNetwork,
           gamma: float, device: torch.device) -> tuple: 

       state      = torch.FloatTensor(state).to(device)
       next_state = torch.FloatTensor(next_state).to(device)
       action     = torch.FloatTensor(action).to(device)
       reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
       done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

       predicted_q_value1 = soft_q_network1(state, action)
       predicted_q_value2 = soft_q_network2(state, action)
       predicted_value    = value_network(state)
       new_action, log_prob, _, _, _ = policy_network.sample(state)

       #TD target
       target_value = target_value_network(next_state)
       target_q_value = reward + (1 - done) * gamma * target_value

       # Updating Q-value Functions
       q_value_loss1 = torch.nn.MSELoss()(predicted_q_value1, target_q_value.detach())
       q_value_loss1.backward()

       q_value_loss2 = torch.nn.MSELoss()(predicted_q_value2, target_q_value.detach())
       q_value_loss2.backward()

       # Updating Value Function
       predicted_new_q_value = torch.min(soft_q_network1(state, new_action), #action follows new policy
                                         soft_q_network2(state, new_action))
       target_value_func = predicted_new_q_value - log_prob

       value_loss = torch.nn.MSELoss()(predicted_value, target_value_func.detach())
       value_loss.backward()

       # Updating Policy Function
       policy_loss = (log_prob - predicted_new_q_value).mean()
       policy_loss.backward()

       return value_loss, q_value_loss1, q_value_loss2, policy_loss


