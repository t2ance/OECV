import json
from abc import abstractmethod


class JSONAble:
    @abstractmethod
    def to_dict(self):
        raise NotImplementedError


class ConfigurationEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, JSONAble):
            return obj.to_dict()
        return obj.__name__


class ConfigurationStore:
    config = None

    @classmethod
    def set_configuration(cls, config):
        cls.config = config

    @classmethod
    def get_configuration(cls):
        return cls.config


if __name__ == '__main__':
    ...
