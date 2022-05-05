import click

from .ensemble import Ensemble
from .definitions import META_ANALYSIS_MODEL, LAB_DATA_MODEL
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@click.command("cli")
@click.argument("target", type=click.Choice(["log_z_av", "PDI"]))
@click.argument("dataset", type=str, default=click.Choice(["lab", "merged"]))
@click.option("--repeats", type=int, default=100)
@click.option("--df", type=click.Path(exists=True), default=None)
@click.option("--tune_steps", type=int, default=200)
@click.option("--train_size", type=float, default=0.7)
def main(target, dataset, repeats, df, tune_steps, train_size):
    dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M")

    if dataset == "merged":
        options = META_ANALYSIS_MODEL
        if df is None:
            # ToDo: use pystow once the data is on Zenod
            df = pd.read_pickle(os.path.join(THIS_DIR, "data", "preprocess_data_merged"))
            df["PDI"] = df["DLS PDI"]
    elif dataset == "lab":
        options = LAB_DATA_MODEL
        if df is None:
            df = pd.read_csv(os.path.join(THIS_DIR, "data", "AllSizes_allRuns.csv"), sep=";")
            df["log_z_av"] = np.log(df["Z-Average [nm]"])
            df["PDI"] = df["Mean PDI"]

    for repeat in range(repeats):

        all_features = options["continuos_features"] + options["cat_features"]
        df_clean = df.dropna(subset=all_features + [target])
        df_clean = df_clean.reset_index()

        ensemble = Ensemble(
            "",
            df_clean,
            cat_features=options["cat_features"],
            seed=np.random.randint(0, 10000),
            continuous_feat=options["continuos_features"],
            train_size=train_size,
            label=target,
        )

        ensemble._tune_hp(tune_steps)
        ensemble.train()

        ensemble.dump_summarized_predict(f"{dt_string}_{dataset}_{target}_{repeat}.pkl")
        with open(f"{dt_string}__{dataset}_{target}_{repeat}_model.pkl", "wb") as handle:
            pickle.dump(ensemble, handle)

        shap_results = ensemble.get_shap_values(df_clean)
        with open(f"{dt_string}_{dataset}_{target}_{repeat}_shap.pkl", "wb") as handle:
            pickle.dump(shap_results, handle)

        with open(f"{dt_string}_{dataset}_{target}_{repeat}_params.pkl", "wb") as handle:
            pickle.dump(ensemble.best_params, handle)


if __name__ == "__main__":
    main()
