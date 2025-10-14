import optuna
import sys
import logging

study_name = "study12ARMA_09_24_25"
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
storage_name = "sqlite:///saved_Optuna/{}.db".format(study_name)

study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
best_row = df.loc[df["value"].idxmax()]
print("\n\n", best_row)