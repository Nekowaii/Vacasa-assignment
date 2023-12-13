from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NUMERICAL_FEATURES = {
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "total_of_special_requests",
    "required_car_parking_spaces",
    "previous_cancellations",
    "is_repeated_guest",
    "agent",
    "company",
}


CATEGORICAL_FEATURES = {
    "hotel",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "arrival_date_month",
    "customer_type",
    "meal",
    "deposit_type",
}


FEATURES = NUMERICAL_FEATURES | CATEGORICAL_FEATURES


def get_selected_features(features: list[str] | None = None) -> dict[str, list]:
    if features and not set(features).issubset(FEATURES):
        raise ValueError("Some features are not recognized as either numerical or categorical.")

    numerical_features = list(NUMERICAL_FEATURES.intersection(features) if features else NUMERICAL_FEATURES)
    categorical_features = list(CATEGORICAL_FEATURES.intersection(features) if features else CATEGORICAL_FEATURES)

    return {"numerical_features": numerical_features, "categorical_features": categorical_features}


def build_preprocessor(numerical_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numerical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="constant"))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, list(numerical_features)),
            ("cat", categorical_pipeline, list(categorical_features)),
        ]
    )

    return preprocessor
