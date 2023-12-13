from pathlib import Path

import pandas as pd


def load_data(file_path: Path, label: str, drop_duplicates: bool = True) -> [pd.DataFrame, pd.Series]:
    data = pd.read_csv(file_path)

    if drop_duplicates:
        data.drop_duplicates(inplace=True)

    return data.drop(columns=label), data[label]


def percentage_change(new_value: float, old_value: float) -> float:
    return (new_value - old_value) / old_value * 100 if old_value != 0 else 0
