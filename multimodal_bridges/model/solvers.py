



def tauleap_step(self, state, delta_t, rates, overflow="wrap"):
    
    delta_n = torch.poisson(rates * delta_t).to(state.time.device) # all jumps
    jump_mask = torch.sum(delta_n, dim=-1).type_as(state.discrete) <= 1 # for categorical data
    
    diff = (
        torch.arange(self.vocab_size, device=state.time.device).view(
            1, 1, self.vocab_size
        )
        - state.discrete[:, :, None]
    )
    net_jumps = torch.sum(delta_n * diff, dim=-1).type_as(state.discrete)

    if overflow == "wrap":
        return (state.discrete + net_jumps * jump_mask) % self.vocab_size

    elif overflow == "clamp":
        state.discrete += net_jumps * jump_mask
        return torch.clamp(state.discrete, min=0, max=self.vocab_size - 1)


def jump_euler_step(self, state, delta_t, rates):
    # off diagonal probs:
    delta_p = (rates * delta_t).clamp(max=1.0) 
    
    # diagonal probs:
    delta_p.scatter_(-1, state.discrete[:, :, None], 0.0)
    delta_p.scatter_(-1, state.discrete[:, :, None], (1.0 - delta_p.sum(dim=-1,keepdim=True)).clamp(min=0.0))

    return Categorical(delta_p).sample() 