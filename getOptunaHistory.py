import optuna
import sys
import logging

study_name = "studyMLP_10_02_25"
# study12ARMA_09_24_25
# studyCGCN_09_25_25
# studyMLP_10_02_25
# studyLSTM_10_27_25

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
storage_name = "sqlite:///saved_Optuna/{}.db".format(study_name)

study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
best_row = df.loc[df["value"].idxmax()]
print("\n\n", best_row)

# print(df.loc[183])