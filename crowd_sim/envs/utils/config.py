from collections import UserDict, Mapping
from pathlib import Path

import toml


def dict_merge(dct, merge_dct):
    """Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if (
            k in dct
            and isinstance(dct[k], dict)
            and isinstance(merge_dct[k], Mapping)
        ):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


class Config(UserDict):
    """
    Config class enables easy loading and getting config entries
    """

    def __init__(self, config) -> None:
        if isinstance(config, str) or isinstance(config, Path):
            print(f"Loading config from {config}")
            config = toml.load(config)
        assert isinstance(
            config, dict
        ), "Config class takes a dict or toml file"
        super().__init__(config)

    # def __getitem__(self, key):
    #     """Return Config rather than dict"""
    #     val = self.data[key]
    #     if isinstance(val, dict):
    #         val = Config(val)
    #     return val

    def __call__(self, *argv):
        """Make class callable to easily get entries"""
        try:
            output = self.data
            for arg in argv:
                if not isinstance(output, dict):
                    print("Too many keys, return the closest entry.")
                    return output
                output = output[arg]
            if isinstance(output, dict):
                output = Config(output)
            return output
        except KeyError as err:
            print(f"Config error {err}")

    def dump(self, config_output):
        """Dump dict to file"""
        with open(config_output, "w") as f:
            toml.dump(self.data, f)

    def merge(self, merge_dct):
        """Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
        updating only top-level keys, dict_merge recurses down into dicts nested
        to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
        ``dct``.
        :param dct: dict onto which the merge is executed
        :param merge_dct: dct merged into dct
        :return: None"""
        for k, v in merge_dct.items():
            if (
                k in self
                and isinstance(self.data[k], dict)
                and isinstance(merge_dct[k], Mapping)
            ):
                dict_merge(self.data[k], merge_dct[k])
            else:
                self[k] = merge_dct[k]
