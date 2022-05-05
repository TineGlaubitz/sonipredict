# -*- coding: utf-8 -*-
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler, StandardScaler
from xgboost import XGBRegressor


def build_model(params, cat_features, continuous_feat):
    pipe = Pipeline(
        [
            (
                "col_trans",
                ColumnTransformer(
                    [
                        # ("scaler", scaler, continuous_feat), # not needed as power transform already applies normalization
                        ("power_transform", PowerTransformer(), continuous_feat),
                        ("one_hot", OneHotEncoder(drop="if_binary"), cat_features),
                    ]
                ),
            ),
            ("booster", XGBRegressor(**params)),
        ]
    )

    return pipe


def build_from_trial(trial, cat_features, continuous_feat):

    params = {
        # "scaler": trial.suggest_categorical("scaler", ["standard", "robust"]),
        "n_estimators": trial.suggest_int("n_estimators", 10, 500),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.4, 0.9),
        "colsample_bylevel": trial.suggest_uniform("colsample_bylevel", 0.4, 0.9),
        # "min_child_weight": trial.suggest_int("min_child_weight", 0, 350),
        "subsample": trial.suggest_uniform("subsample", 0.4, 0.9),
        # "gamma": trial.suggest_uniform("gamma", 0, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 60),
    }

    return build_model(params, cat_features, continuous_feat)
