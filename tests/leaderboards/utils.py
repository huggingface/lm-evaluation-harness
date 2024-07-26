import json
import os
from typing import Any, Dict, Iterator, List


def filter_dict(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    """
    Filter a dictionary to include only the keys that contain the specified substring.

    Args:
        d (Dict): The dictionary to filter.
        key (str): The substring to filter the dictionary keys by.

    Returns:
        Dict: A dictionary containing only the keys that contain the specified substring.
    """
    return {k: v for k, v in d.items() if key in k}


class ConfigParser:
    """
    A class to parse a dictionary object into an attribute-based object.

    The `ConfigParser` class allows accessing dictionary keys as attributes and
    is designed to be used with dictionaries. It supports nested dictionaries and lists,
    recursively converting them into `ConfigParser` objects.

    Attributes:
        dict_obj (Dict): The dictionary object to parse.

    Example:
        config_dict = {
            "transformers_version": "4.42.3",
            "params": {
                "model": "hf",
                "batch_size": 1
            },
        }

        config = ConfigParser(config_dict)

        # Accessing attributes
        print(config.transformers_version)
        >>> 4.42.3
        print(config.params.model)
        >>> hf
    """

    def __init__(self, dict_obj):
        """
        Initialize the ConfigParser object with the given dictionary object.

        Args:
            dict_obj (Dict): The dictionary object to parse.
        """
        for key, value in dict_obj.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigParser(value))
            elif isinstance(value, list):
                setattr(
                    self,
                    key,
                    [
                        ConfigParser(item) if isinstance(item, dict) else item
                        for item in value
                    ],
                )
            else:
                setattr(self, key, value)

    def __getattr__(self, name) -> Any:
        """Get the value of the attribute from the ConfigParser object."""
        raise AttributeError(f"Attribute '{name}' is not defined in the config file.")

    def __getitem__(self, key) -> Any:
        """Get the value of the key from the ConfigParser object."""
        return self.__dict__[key]

    def __repr__(self) -> str:
        """Return the string representation of the ConfigParser object."""
        return f"ConfigParser({self.__dict__})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the ConfigParser object with nested objects into a dictionary."""

        def _convert_to_dict(obj):
            if isinstance(obj, ConfigParser):
                return {key: _convert_to_dict(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [_convert_to_dict(item) for item in obj]
            else:
                return obj

        return _convert_to_dict(self)

    def keys(self) -> Iterator:
        """Return the keys of the ConfigParser object."""
        if hasattr(self, "__dict__"):
            return self.__dict__.keys()
        else:
            raise TypeError("Object is not a dictionary.")

    def items(self) -> Iterator:
        """Return an iterator of the ConfigParser object's items."""
        if hasattr(self, "__dict__"):
            return iter(self.__dict__.items())
        else:
            raise TypeError("Object is not a dictionary.")


def load_all_configs(device: str = "cpu") -> List[Dict]:
    """
    Load all configuration files that include the specified device in their name.

    This function reads configuration files from the 'tests/leaderboards/testconfigs' directory.
    It filters the files to include only those that match the given device (e.g., "cpu" or "gpu").

    Args:
        device (str): The device to filter the configuration files by.
            Valid values are "cpu" or "gpu".

    Returns:
        List[Dict]: A list of dictionaries containing the configuration data from the files.
    """
    valid_devices = {"cpu", "gpu"}
    if device not in valid_devices:
        raise ValueError(f"Invalid device {device}. Must be one of {valid_devices}.")

    configs_dir = os.path.join("tests", "leaderboards", "testconfigs")
    config_files = [
        f for f in os.listdir(configs_dir) if device in f and f.endswith(".json")
    ]

    data_list = []
    for filename in config_files:
        filepath = os.path.join(configs_dir, filename)
        with open(filepath, "r") as f:
            data_list.append(json.load(f))
    return data_list
