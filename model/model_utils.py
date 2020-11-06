"""
    utilities to register models in a global list
"""

global mapping
mapping = {}

# decorator
def register(name):
    """
    Decorator: register a network builder function in a dictionary 'mapping' by name: func

    Args:
        name: (str) network name; used as arguments passed to the decorator


    Note:
    - a network builder is a function that takes **kwargs arguments and returns a network object
        - i.e., call this function by MyNetwork = network_func(**kwargs)

    To register a new network builder:
    in another_network.py file, do:

    from model_utils import register
    @register(name='new_network_builder')
    def new_network_builder(**network_kwargs):
        ...
        return network_function_or_class_obj

    """
    def _decorator(func):
        """
        inner decorator function
        - no need to modify behaviors so no wrappers
        """
        mapping[name] = func
        return func
    return _decorator
