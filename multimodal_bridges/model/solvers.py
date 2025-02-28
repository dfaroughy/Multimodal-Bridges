import torch
from torch.distributions import Categorical

class DiscreteSolverStep:

    def __init__(self, config):
        self.config = config

    def tauleap(self, state, rates, delta_t, overflow="wrap"):

        state.discrete = state.discrete.squeeze(-1)
        delta_n = torch.poisson(rates * delta_t).to(state.time.device) # all jumps
        jump_mask = torch.sum(delta_n, dim=-1).type_as(state.discrete) <= 1 # for categorical data
        
        diff = (
            torch.arange(self.vocab_size, device=state.time.device).view(
                1, 1, self.vocab_size
            )
            - state.discrete[:, :, None]
        )
        net_jumps = torch.sum(delta_n * diff, dim=-1).type_as(state.discrete)

        # take step

        if overflow == "wrap":
            state.discrete = (state.discrete + net_jumps * jump_mask) % self.vocab_size
            state.discrete = state.discrete.unsqueeze(-1)
            return state

        elif overflow == "clamp":
            state.discrete += net_jumps * jump_mask
            state.discrete = torch.clamp(state.discrete, min=0, max=self.vocab_size - 1)
            state.discrete = state.discrete.unsqueeze(-1)
            return state

    def euler(self, state, rates, delta_t):

        # rates
        # rates = self.rate(state, heads)

        # off diagonal probs:
        state.discrete = state.discrete.squeeze(-1)
        delta_p = (rates * delta_t).clamp(max=1.0) 
        
        # diagonal probs:
        delta_p.scatter_(-1, state.discrete[:, :, None], 0.0)
        delta_p.scatter_(-1, state.discrete[:, :, None], (1.0 - delta_p.sum(dim=-1,keepdim=True)).clamp(min=0.0))

        state.discrete = Categorical(delta_p).sample()
        state.discrete = state.discrete.unsqueeze(-1)
        return state