
__all__ = ['SetattrInitMixin']

class SetattrInitMixin(object):
    def __init__(self, **kwargs):
        ''' see class docstring for relevant attributes '''
        for k, v in kwargs.items():
            setattr(self, k, v)

