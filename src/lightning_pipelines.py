
from src.lightning import CounterfactualRegressionLightning, DragonNetLightning
from src.lightning import get_checkpoint_callback, get_log_dir_path, get_trainer, get_logger
from src.dataset import XYTDataModule

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
    datamodule = XYTDataModule(treatment_col=treatment_col,
                                outcome_col=outcome_col,
                                input_features=input_features,
                                dataset_name=dataset_name,
                                raw_data=raw_data)

    # set up model
    model = CounterfactualRegressionLightning(input_features=input_features,
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


def DragonNet_training_pipeline(**kwargs):
    # read out kwargs
    treatment_col = kwargs.get('treatment_col')
    outcome_col = kwargs.get('outcome_col')
    input_features = kwargs.get('input_features')
    alpha = kwargs.get('alpha')
    beta = kwargs.get('beta')
    dataset_name = kwargs.get('dataset_name')
    model_name = kwargs.get('model_name')
    wandb_kwargs = kwargs.get('wandb_kwargs', {})
    raw_data = kwargs.get('raw_data')
    target_reg = kwargs.get('target_reg')
    
    # set up data module
    datamodule = XYTDataModule(treatment_col=treatment_col,
                                outcome_col=outcome_col,
                                input_features=input_features,
                                dataset_name=dataset_name,
                                raw_data=raw_data)

    # set up model
    model = DragonNetLightning(input_features=input_features,
                                alpha=alpha,
                                beta=beta,
                                target_reg=target_reg)

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