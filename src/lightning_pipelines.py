from src.lightning import CounterfactualRegressionLightning, get_checkpoint_callback, get_log_dir_path, get_trainer, get_logger
from src.dataset import CFRDataModule

def CFR_training_pipeline(**kwargs):
    # read out kwargs
    treatment_col = kwargs.get('treatment_col')
    outcome_col = kwargs.get('outcome_col')
    input_features = kwargs.get('input_features')
    alpha = kwargs.get('alpha')
    dataset_name = kwargs.get('dataset_name')
    outcome_type = kwargs.get('outcome_type')
    model_name = kwargs.get('model_name')
    wandb_kwargs = kwargs.get('wandb_kwargs', {})
    raw_data = kwargs.get('raw_data')
    
    # set up data module
    datamodule = CFRDataModule(treatment_col=treatment_col,
                                outcome_col=outcome_col,
                                input_features=input_features,
                                dataset_name=dataset_name,
                                raw_data=raw_data)

    # set up model
    model = CounterfactualRegressionLightning(input_features=input_features,
                                              treatment_col=treatment_col,
                                              outcome_col=outcome_col,
                                              alpha=alpha,
                                              outcome_type=outcome_type)

    # get log dir
    log_dir_path = get_log_dir_path(model_name)

    # get checkpoint callback
    checkpoint_callback = get_checkpoint_callback(model_name, log_dir_path)

    # get logger
    logger = get_logger(model_name=model_name, **wandb_kwargs)

    # get trainer
    trainer = get_trainer(model_name, checkpoint_callback, logger=logger)

    print("Training model")
    trainer.fit(model, datamodule)
    
    return {'trainer': trainer, 'model':model, 'datamodule': datamodule}