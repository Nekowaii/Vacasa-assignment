import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
from sklearn.metrics import check_scoring
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from argument_parser import ModelsAction, ModelParamsAction
from preprocessing import build_preprocessor, get_selected_features, FEATURES
from promotion import models_comparison_report, is_new_promoted_model, save_promoted_model
from utils import load_data

logging.basicConfig(level=logging.INFO)

SCORERS = ["roc_auc", "recall", "precision", "f1", "accuracy"]
MODEL_DIR = Path("models")


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-i",
        "--infile",
        help="File to be processed",
        type=Path,
        default=Path(__file__).absolute().parent / "data/train/hotel_bookings.csv",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Specify the training model to use. Available models: %(choices)s",
        action=ModelsAction,
        required=True,
        metavar="MODEL",
    )
    parser.add_argument(
        "-p", "--params", nargs="*", action=ModelParamsAction, default={}, help="Model parameters.", metavar="KEY=VALUE"
    )
    parser.add_argument(
        "-f",
        "--features",
        nargs="*",
        help=f"Available features (Default ALL): \n{'\n'.join(FEATURES)}",
        choices=FEATURES,
        metavar="NAME",
    )
    parser.add_argument(
        "-l", "--label", nargs="*", help=f"Label for classification. Default is %(default)s", default="is_canceled"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        choices=range(0, 101),
        metavar="[0-100]",
        help="Parameterize the %(metavar)s%% threshold increase to promote a model. Default is %(default)s%%.",
        default=10,
    )
    parser.add_argument(
        "-s",
        "--scoring",
        choices=SCORERS,
        help="Parameterize which evaluation metric can be used. Default is %(default)s.",
        default="roc_auc",
    )
    parser.add_argument("-с", "--clean", action="store_true", help="Clean all promoted models during training.")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.clean and MODEL_DIR.exists():
        logging.info("Cleaning promoted models...")
        shutil.rmtree(MODEL_DIR)

    X, y = load_data(args.infile, args.label)

    selected_features = get_selected_features(args.features)
    preprocessor = build_preprocessor(**selected_features)
    classifier = args.model(**args.params)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

    logging.info("Evaluating model...")
    scores = evaluate_model(X, y, pipeline, SCORERS)
    models_comparison_report(scores, args.scoring)

    if is_new_promoted_model(scores, args.scoring, args.threshold):
        logging.info("Promotion detected. Training new model ...")
        pipeline.fit(X, y)
        save_promoted_model(
            pipeline, scores=scores, features=selected_features, params=args.params, model=args.model.__name__
        )
        logging.info("New model promoted.")
    else:
        logging.info(f"New model didn't meet the threshold for promotion.")


def evaluate_model(X, y, pipeline: Pipeline, scorers: list[str]) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    results = cross_validate(
        pipeline, X, y, cv=cv, scoring={scoring: check_scoring(pipeline, scoring=scoring) for scoring in scorers}
    )

    scores = {
        name.removeprefix("test_"): np.mean(values) for name, values in results.items() if name.startswith("test_")
    }
    return scores


if __name__ == "__main__":
    main()
