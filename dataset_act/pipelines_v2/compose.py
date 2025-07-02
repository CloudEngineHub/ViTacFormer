# import collections

# from ...utils import instantiate_from_config
# from ..builder import PIPELINES, build_from_cfg


# class Compose:
#     """Compose multiple transforms sequentially. The pipeline registry of
#     mmdet3d separates with mmdet, however, sometimes we may need to use mmdet's
#     pipeline. So the class is rewritten to be able to use pipelines from both
#     mmdet3d and mmdet.

#     Args:
#         transforms (Sequence[dict | callable]): Sequence of transform object or
#             config dict to be composed.
#     """

#     def __init__(self, transforms):
#         assert isinstance(transforms, collections.abc.Sequence), type(transforms)
#         self.transforms = []
#         for transform in transforms:
#             if isinstance(transform, dict):
#                 if "type" not in transform and "target" in transform:
#                     transform = instantiate_from_config(transform)
#                 else:
#                     _, key = PIPELINES.split_scope_key(transform["type"])
#                     if key in PIPELINES._module_dict.keys():
#                         transform = build_from_cfg(transform, PIPELINES)
#                     else:
#                         raise NotImplementedError(
#                             f"key {key} not found in regristry PIPELINES"
#                         )
#                         # transform = build_from_cfg(transform, MMDET_PIPELINES)
#                 self.transforms.append(transform)
#             elif callable(transform):
#                 self.transforms.append(transform)
#             else:
#                 raise TypeError("transform must be callable or a dict")

#     def __call__(self, data):
#         """Call function to apply transforms sequentially.

#         Args:
#             data (dict): A result dict contains the data to transform.

#         Returns:
#            dict: Transformed data.
#         """

#         for t in self.transforms:
#             data = t(data)
#             if data is None:
#                 return None
#         return data

#     def __repr__(self):
#         format_string = self.__class__.__name__ + "("
#         for t in self.transforms:
#             format_string += "\n"
#             format_string += f"    {t}"
#         format_string += "\n)"
#         return format_string

import importlib
import collections

# def instantiate_from_config(config):
#     """
#     Instantiate a class or function from config.

#     Args:
#         config (dict): A config dictionary with a 'target' key.
#             Example:
#                 {
#                     "target": "mymodule.MyClass",
#                     "params": {"arg1": 1, "arg2": 2}
#                 }

#     Returns:
#         Instantiated object.
#     """
#     assert "target" in config, "Config must have a 'target' field"
#     target = config["target"]
#     params = config.get("params", {})

#     # Dynamically import module and get class/function
#     module_path, cls_name = target.rsplit(".", 1)
#     module = importlib.import_module(module_path)
#     cls_or_func = getattr(module, cls_name)

#     return cls_or_func(**params)

def instantiate_from_config(config):
    """
    Instantiate a class or function from config.
    Supports both:
    - {"target": "xxx.Class", "params": {...}}
    - {"target": "xxx.Class", "arg1": ..., "arg2": ...}
    """
    assert "target" in config, "Config must have a 'target' field"
    target = config["target"]

    # Try both param styles
    params = dict(config)  # shallow copy
    params.pop("target")
    params.pop("params", None)  # prevent conflict

    # If "params" is used, update with that
    if "params" in config:
        params.update(config["params"])

    module_path, cls_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls_or_func = getattr(module, cls_name)

    return cls_or_func(**params)


class Compose:
    """
    Compose multiple transforms sequentially.

    Supports:
    - Callable transforms
    - Config dicts with "target" and optional "params"
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence), type(transforms)
        self.transforms = []
        for transform in transforms:
            # print(transform)
            if isinstance(transform, dict) and "target" in transform:
                transform = instantiate_from_config(transform)
            elif not callable(transform):
                raise TypeError(f"Transform must be callable or a config dict with 'target'. Got {type(transform)}")
            self.transforms.append(transform)

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data
