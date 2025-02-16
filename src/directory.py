import os

# set data dir
data_dir = '../data'
log_dir = '../logs'

for _dir in [log_dir]:
    if not os.path.isdir(_dir):
        os.makedirs(_dir)

dataset_paths = dict(
    faisalabad=os.path.join(data_dir, "heart_failure_clinical_records_dataset.csv"),
    unos=os.path.join(data_dir, "UNOS_train.csv"),
    ihdp=os.path.join(data_dir, "ihdp_project2.csv"),
    ist_corrected=os.path.join(data_dir, "IST/IST_corrected.txt"),
    ist_original=os.path.join(data_dir, "IST/IST_observational.csv")

)

deep_survival_model_path = os.path.join(log_dir, 'deep_survival', 'best_model.pth')