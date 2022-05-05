# -*- coding: utf-8 -*-
import pickle
import random
from copy import deepcopy
from typing import Union, List
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna import Trial
from sklearn.dummy import DummyRegressor
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .utils import build_from_trial, build_model
import shap


class Ensemble:
    """Builds an ensemble classifier for repeated measurements"""

    def __init__(
        self,
        base_model: object,
        df: pd.DataFrame,
        train_size: float = 0.7,
        seed: int = 12345,
        cat_features: Union[List[str], None] = None,
        continuous_feat: Union[List[str], None] = None,
        label: Union[str, None] = None,
    ):
        self.df = df.sample(len(df))  # shuffle the df
        self.seed = seed
        self.label = label
        self.features = continuous_feat + cat_features
        self.cat_features = cat_features
        self.continuous_feat = continuous_feat
        unique_comb = df[self.features].drop_duplicates()
        self.unique_combinations = unique_comb.values
        self._labels = self.df[self.label].iloc[unique_comb.index]
        self.n_estimators = len(
            df[(df[self.features].values == self.unique_combinations[-1, :]).all(axis=1)]
        )
        self.estimators = [deepcopy(base_model) for n in range(self.n_estimators)]
        self.train_size = train_size
        self._get_train_test_idx()
        self._dummy_metrics = None
        self._trained = False

    def dump(self, filename):
        with open(filename, "wb") as handle:
            pickle.dump(self, handle)

    def _tune_study_parallel_coord(self):
        return optuna.visualization.plot_parallel_coordinate(self.study)

    def _tune_study_history_plot(self):
        return optuna.visualization.plot_optimization_history(self.study)

    def _get_train_test_idx(self):
        idx = range(len(self.unique_combinations))
        size = np.quantile(self._labels, 0.6)
        stratification_series = [1 if s > size else 0 for s in self._labels]
        self._train_idx, self._test_idx = train_test_split(
            idx,
            train_size=self.train_size,
            stratify=stratification_series,
            shuffle=True,
            random_state=self.seed,
        )

    def _tune_hp(self, trials=10):
        study = optuna.create_study(direction="minimize")

        x, y = self._get_dataset(1)

        def objective(trial: Trial, fastCheck=True):
            size = np.mean(self._labels)
            stratification_series = [1 if s > size else 0 for s in y]
            kfold = StratifiedKFold(n_splits=10)

            metrics = []
            for train_index, test_index in kfold.split(x, stratification_series):
                train_data = x.iloc[train_index], y[train_index]
                valid_data = x.iloc[test_index], y[test_index]
                pipe = build_from_trial(trial, self.cat_features, self.continuous_feat)
                pipe = pipe.fit(train_data[0], train_data[1])

                valid_predict = pipe.predict(valid_data[0])
                metrics.append(mean_absolute_error(valid_data[1], valid_predict))

            return np.mean(metrics)

        study.optimize(objective, n_trials=trials)

        best_params = study.best_trial.params
        self.best_params = best_params

        pipe = build_model(best_params, self.cat_features, self.continuous_feat)

        self.estimators = [deepcopy(pipe) for n in range(self.n_estimators)]

        self.study = study

    def _get_dataset(self, i, split="train"):
        data = []

        if split == "train":
            idx = self._train_idx
        elif split == "test":
            idx = self._test_idx

        relevant_conditions = self.unique_combinations[idx]

        for condition in relevant_conditions:
            t_df = self.df[(self.df[self.features] == condition).all(axis=1)]
            t_df.reset_index()
            try:
                data.append(t_df.values[i])
            except Exception:
                data.append(t_df.values[0])

        temp_df = pd.DataFrame(data, columns=self.df.columns)

        return temp_df[self.features], temp_df[self.label].values

    def _train(self, y_scramble=False):
        for i, estimator in enumerate(self.estimators):
            x, y = self._get_dataset(i, "train")
            if y_scramble:
                random.shuffle(y)
            estimator.fit(x, y)

        if y_scramble:
            self._trained = False
        else:
            self._trained = True

    def get_shap_values(self, df):
        explainers = [shap.TreeExplainer(estimator.steps[-1][1]) for estimator in self.estimators]
        sub_pip = self.estimators[0].steps[:-1]
        X = sub_pip[0][1].transform(df[self.features])
        shap_values = []
        for explainer in explainers:
            shap_values.append(explainer.shap_values(X))
        return shap_values, np.mean(shap_values, axis=0)

    def _train_all(self):
        for i, estimator in enumerate(self.estimators):
            x_train, y_train = self._get_dataset(i, "train")
            x_test, y_test = self._get_dataset(i, "test")
            x = pd.concat([x_train, x_test])
            y = np.vstack([y_train.reshape(-1, 1), y_test.reshape(-1, 1)])
            estimator.fit(x, y)

    def _get_metric_dict(self, pred, true):
        return {
            "mae": mean_absolute_error(true, pred),
            "mse": mean_squared_error(true, pred),
            "r2": r2_score(true, pred),
            "max_error": max_error(true, pred),
        }

    def _dummy_scores(self):
        # ToDo: refactor to get errorbars on metrics
        if self._dummy_metrics is None:
            dummy_mean = DummyRegressor(strategy="mean")
            dummy_median = DummyRegressor(strategy="median")

            dummy_models = [
                ("mean", dummy_mean),
                ("median", dummy_median),
            ]

            metrics = {}

            for name, model in dummy_models:

                predictions_train = []
                truth_train = []

                predictions_test = []
                truth_test = []

                for i in range(len(self.estimators)):
                    x_train, y_train = self._get_dataset(i, "train")
                    x_test, y_test = self._get_dataset(i, "test")
                    model.fit(x_train, y_train)
                    train_pred = model.predict(x_train)
                    test_pred = model.predict(x_test)

                    predictions_train.append(train_pred)
                    predictions_test.append(test_pred)

                    truth_train.append(y_train)
                    truth_test.append(y_test)

                summary_dict_train = self._summarize(predictions_train, truth_train)
                summary_dict_test = self._summarize(predictions_test, truth_test)

                metrics[name] = {}

                metrics[name]["train"] = self._get_metric_dict(
                    summary_dict_train["labels"]["mean"],
                    summary_dict_train["prediction"]["mean"],
                )
                metrics[name]["test"] = self._get_metric_dict(
                    summary_dict_test["labels"]["mean"],
                    summary_dict_test["prediction"]["mean"],
                )

            self._dummy_metrics = metrics

        return self._dummy_metrics

    def _summarize(self, prediction, truth):
        return {
            "prediction": {
                "mean": np.mean(prediction, axis=0),
                "std": np.std(prediction, axis=0),
            },
            "labels": {"mean": np.mean(truth, axis=0), "std": np.std(truth, axis=0)},
        }

    @property
    def dummy_scores(self):
        return self._dummy_scores()

    def _predict(self, split="test"):
        predicted = []
        ground_truth = []
        for i in range(self.n_estimators):
            x, y = self._get_dataset(i, split)
            pred = self.estimators[i].predict(x)
            predicted.append(pred)
            ground_truth.append(y)

        return predicted, ground_truth

    def predict(self, X):
        assert self._trained
        predictions = []
        for estimator in self.estimators:
            predictions.append(estimator.predict(X.reshape(-1, len(self.features))))

        return {
            "predictions": predictions,
            "mean_prediction": np.mean(predictions, axis=0),
            "std_predictions": np.std(predictions, axis=0),
        }

    def _summarized_predict(self, split="test"):
        prediction, truth = self._predict(split)

        return self._summarize(prediction, truth)

    @property
    def scores(self):
        train_pred = self._summarized_predict("train")
        test_pred = self._summarized_predict("test")

        scores = {}
        scores["train"] = {}
        scores["test"] = {}

        scores["train"] = self._get_metric_dict(
            train_pred["labels"]["mean"], train_pred["prediction"]["mean"]
        )
        scores["test"] = self._get_metric_dict(
            test_pred["labels"]["mean"], test_pred["prediction"]["mean"]
        )

        return scores

    @property
    def y_scramble_scores(self):
        self._train(y_scramble=True)
        train_pred = self._summarized_predict("train")
        test_pred = self._summarized_predict("test")

        scores = {}
        scores["train"] = {}
        scores["test"] = {}

        scores["train"] = self._get_metric_dict(
            train_pred["labels"]["mean"], train_pred["prediction"]["mean"]
        )
        scores["test"] = self._get_metric_dict(
            test_pred["labels"]["mean"], test_pred["prediction"]["mean"]
        )

        return scores

    def dump_summarized_predict(self, outname: str = None):
        train_pred = self._summarized_predict("train")
        test_pred = self._summarized_predict("test")
        dump_dict = {}
        dump_dict["train"] = train_pred
        dump_dict["test"] = test_pred

        with open(outname, "wb") as handle:
            pickle.dump(dump_dict, handle)

    def _parity_plot(self, errorbar: bool = True):
        fig, ax = plt.subplots(1, 2, sharex="all", sharey="all")

        train_pred = self._summarized_predict("train")
        test_pred = self._summarized_predict("test")

        for a in ax:
            a.set_xlabel(r"$y$")

        ax[0].set_ylabel(r"$\hat{y}$")
        ax[0].set_title("train")
        ax[1].set_title("test")

        if (self.n_estimators > 1) and errorbar:
            ax[0].errorbar(
                train_pred["labels"]["mean"],
                train_pred["prediction"]["mean"],
                train_pred["prediction"]["std"],
                train_pred["labels"]["std"],
                "none",
            )

            ax[1].errorbar(
                test_pred["labels"]["mean"],
                test_pred["prediction"]["mean"],
                test_pred["prediction"]["std"],
                test_pred["labels"]["std"],
                "none",
            )
        else:
            ax[0].scatter(train_pred["labels"]["mean"], train_pred["prediction"]["mean"], s=4)

            ax[1].scatter(test_pred["labels"]["mean"], test_pred["prediction"]["mean"], s=4)

        ax[0].plot(ax[0].get_xlim(), ax[0].get_ylim(), ls="--", c="black")
        ax[1].plot(ax[1].get_xlim(), ax[1].get_ylim(), ls="--", c="black")

        fig.tight_layout()

    def parity_plot(self, errorbar: bool = True):
        self._parity_plot(errorbar)

    def train(self):
        self._train()
