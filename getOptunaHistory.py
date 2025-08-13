import optuna

study_name = "studyARMA_posweight507_15_25"  # Unique identifier of the study.
storage_name = "sqlite:///saved_Optuna/{}.db".format(study_name)

study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
print(df)