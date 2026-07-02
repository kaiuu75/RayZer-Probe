from .linear_probe import LinearDepthProbe
from .mlp_probe import MLPDepthProbe

PROBE_CHOICES = ['linear', 'mlp']

def get_probe(name, input_dim=768):
    """Factory function to create a probe by name.
    
    Args:
        name: One of 'linear' or 'mlp'.
        input_dim: Dimension of the input token features.
    
    Returns:
        An nn.Module probe instance.
    """
    if name == 'linear':
        return LinearDepthProbe(input_dim=input_dim)
    elif name == 'mlp':
        return MLPDepthProbe(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown probe type: '{name}'. Choose from {PROBE_CHOICES}")
