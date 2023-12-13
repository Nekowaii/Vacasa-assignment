import argparse
import importlib


class ModelsAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        self.classifiers = {
            "XGBClassifier": "xgboost",
            "LGBMClassifier": "lightgbm",
            "CatBoostClassifier": "catboost",
            "AdaBoostClassifier": "sklearn.ensemble",
            "GradientBoostingClassifier": "sklearn.ensemble",
            "ExtraTreesClassifier": "sklearn.ensemble",
            "RandomForestClassifier": "sklearn.ensemble",
            "DecisionTreeClassifier": "sklearn.tree",
        }

        super().__init__(option_strings, dest, choices=self.classifiers, **kwargs)

    def _dynamic_import(self, model_name):
        class_path = self.classifiers[model_name]
        module = importlib.import_module(class_path)
        return getattr(module, model_name)

    def _get_model(self, name):
        if name in self.classifiers:
            class_path = self.classifiers[name]
            module = importlib.import_module(class_path)
            return getattr(module, name)

        raise ValueError(f"Model {name} not found.")

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self._get_model(values))


class ModelParamsAction(argparse.Action):
    @staticmethod
    def _convert(value):
        for func in [int, float, bool]:
            try:
                return func(value)
            except ValueError:
                continue

        return value

    def __call__(self, parser, namespace, values, option_string=None):
        params_dict = dict()
        for value in values:
            key, val = map(str.strip, value.split("="))
            params_dict[key] = self._convert(val)
        setattr(namespace, self.dest, params_dict)
