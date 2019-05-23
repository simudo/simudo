import os

def generate_base32_token(length):
    base32_lut = 'ybndrfg8ejkmcpqxot1uwisza345h769'
    return ''.join(base32_lut[i&31] for i in os.urandom(length))

class BaseAnnexRef(yaml.YAMLObject):
    yaml_tag = '!BaseExtRef'
    filename_suffix = ''

    def __init__(self, repository, key=None):
        self.repository = repository
        if key is None:
            while True:
                self.key = 'X' + generate_base32_token(40)
                if not self.exists():
                    break
        else:
            self.key = key

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.key)

    def __call__(self):
        return self.value

    @property
    def value(self):
        raise NotImplementedError()

    @value.setter
    def value(self, value):
        raise NotImplementedError()

    @property
    def filename(self):
        return self.repository.get_filename(self)

    def exists(self):
        return os.path.exists(self.filename)

    @classmethod
    def init_repository(cls, repository):
        pass

    def copy_from(self, other):
        '''stupid copy, override if possible'''
        self.value = other.value

    @classmethod
    def to_yaml(cls, dumper, data):
        key = data.key

        dumper_repository = dumper.yamlannex_repository
        if dumper_repository != data.repository:
            new_data = cls(dumper_repository, key)
            new_data.copy_from(data)
            data = new_data

        return dumper.represent_scalar(self.yaml_tag, data.key)

    @classmethod
    def from_yaml(cls, loader, node):
        key = loader.construct_scalar(node)
        return cls(loader.yamlannex_repository, value)

    @classmethod
    def can_handle(cls, value):
        return False

class DolfinHDF5Ref(yaml.YAMLObject):
    yaml_tag = '!DolfinHDF5Ref'
    filename_suffix = '.h5'

    @property
    def value(self):
        f = dolfin.HDF5File(comm, self.filename, 'r')
        return f.read()

    @value.setter
    def value(self, value):
        raise NotImplementedError()

    @classmethod
    def can_handle(cls, value):
        if isinstance(value, dolfin.Mesh):
            return True
        return False

class YARepository(object):
    serializers = [DolfinHDF5Ref]

    def __init__(self, filename_prefix):
        pass

    def get_filename(self, ref):
        return self.filename_prefix + ref.key + ref.suffix

    def ref(self, value):
        for s in self.serializers:
            if s.can_handle(value):
                ref = s(self)
                ref.value = value
                return ref
        else:
            raise TypeError()

    @property
    def meta(self):
        pass

    @meta.setter
    def meta(self, value):
        pass

class BaseYADumper(yaml.BaseDumper):
    def __init__(self, *args, hdfstore, **kwargs):
        self.hdfstore = hdfstore
        self.pandas_seen = {}
        super().__init__(*args, **kwargs)
        self.add_representer(pandas.Series,    self.pandas_representer)
        self.add_representer(pandas.DataFrame, self.pandas_representer)
    def add_representer_from_class(self, cls):
        self.add_constructor(cls.yaml_tag, cls.from_yaml)
    def pandas_representer(self, dumper, data):
        d = self.pandas_seen
        k = id(data)
        if k in d:
            data = d[k]
        else:
            data = d[k] = pandas_to_hdfref(self.hdfstore, data)
        return data.to_yaml(dumper, data)

class BaseYALoader(yaml.BaseLoader):
    def __init__(self, *args, hdfstore, **kwargs):
        self.hdfstore = hdfstore
        super().__init__(*args, **kwargs)
        self.add_constructor_from_class(HDFRef)
    def add_constructor_from_class(self, cls):
        self.add_constructor(cls.yaml_tag, cls.from_yaml)

class YADumper(BaseHDumper, yaml.Dumper):
    pass

class YALoader(BaseHLoader, yaml.Loader):
    pass

    
to_ref()

ExtRef.to_reference(df)

dumper.to_ref(df)

asdf.yaml.annex/Rasdflkajweflkj.h5

