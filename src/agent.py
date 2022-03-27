import torch
import torch.nn.functional as F

class ValueNetwork(torch.nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, init_w: float) -> None:
        super().__init__()
        self.feature_extractor = None
        self.in_linear = torch.nn.Linear(in_features = state_dim,
                                         out_features = hidden_dim)
        self.int_linear = torch.nn.Linear(in_features = hidden_dim,
                                         out_features = hidden_dim)
        self.logits = torch.nn.Linear(in_features = hidden_dim,
                                         out_features = 1)
        
        self.logits.weight.data.uniform_(-init_w, init_w)
        self.logits.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:

        x = F.relu(
                    self.in_linear(state) if self.feature_extractor is None\
                                          else self.feature_extractor(state)
                  )
        x = F.relu(self.int_linear(x))
        x = self.logits(x)
        return x

class SoftQvalueNetwork(torch.nn.Module):
    def __init__(self, n_inputs: int, action_dim: int, hidden_dim: int, init_w: float) -> None:
        super().__init__()
        self.feature_extractor = None
        self.in_linear = torch.nn.Linear(in_features = action_dim + n_inputs,
                                         out_features = hidden_dim)
        self.int_linear = torch.nn.Linear(in_features = hidden_dim,
                                         out_features = hidden_dim)
        self.logits = torch.nn.Linear(in_features = hidden_dim,
                                         out_features = 1)
        
        self.logits.weight.data.uniform_(-init_w, init_w)
        self.logits.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:

        x = F.relu(
                    self.in_linear(state) if self.feature_extractor is None\
                                          else self.feature_extractor(state)
                  )
        x = F.relu(self.int_linear(x))
        x = self.logits(x)
        return 



class PolicyNetwork(torch.nn.Module):
    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int,
                       init_w: float, log_std_min: float, log_std_max: float):
        super().__init__()
        self.device = self.device()
        self.feature_extractor = None
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.in_linear = torch.nn.Linear(num_inputs, hidden_dim)
        self.int_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = torch.nn.Linear(hidden_dim, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = torch.nn.Linear(hidden_dim, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    @property
    def device(self): return next(self.parameters()).device

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(
                    self.in_linear(state) if self.feature_extractor is None\
                                          else self.feature_extractor(state)
                  )
        x = F.relu(self.int_linear(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, epsilon: float=1e-6) -> tuple:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(0, 1)
        z      = normal.sample()
        action = torch.tanh(mean+ std*z.to(self.device))
        log_prob = torch.distributions.Normal(mean, std).log_prob(mean+ std*z.to(self.device))\
                                             - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        state = state.unsqueeze(0)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(0, 1)
        z      = normal.sample().to(self.device)
        action = torch.tanh(mean + std * z)
        
        action  = action.detach().cpu()
        return action[0]


  