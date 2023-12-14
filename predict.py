import argparse
import logging
from pathlib import Path

from sklearn.metrics import classification_report

from promotion import load_promoted_model
from utils import load_data

logging.basicConfig(level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-i",
        "--infile",
        help="Input file for prediction (RAW)",
        type=Path,
        required=True,  # I assume I'll receive test data on interview
    )
    parser.add_argument(
        "-l", "--label", nargs="*", help=f"Label for classification. Default is %(default)s", default="is_canceled"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    logging.info("Loading data...")
    X_test, y_test = load_data(args.infile, args.label)

    logging.info("Loading model...")
    model, _ = load_promoted_model()

    logging.info("Predicting...")
    y_pred = model.predict(X_test)

    logging.info(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
