"""
    Return a network builder function
"""

from model.model_utils import mapping
from model import resnet

def get_network_builder(name):
    """
    Returns the network builer function by name

    Args:
        name: (str) name of network builder

    Returns:
        (a customized network object) a network object that is customized by **kwarg arguments

    Note:
    - call this function by: MyNetwork = get_network_builder(name)(**kwargs)
    - use the return of this function by: network_output = MyNetwork(network_input)
    """

    if callable(name):
        # if name is a callable function, return directly
        return name
    elif name in mapping.keys():
        # if name is str & registered as a network builder
        return mapping[name]
    else:
        # raise an error otherwise
        raise ValueError('Unknown network type: {}'.format(name))


if __name__ == '__main__':
    print(get_network_builder('resnet18')())
