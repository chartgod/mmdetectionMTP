import os
import subprocess
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import hpbandster.optimizers as optimizers
from hpbandster.core.worker import Worker

from mmengine.config import Config

# Log file path for all BOHB trials
all_trials_log_file_path = '/home/lsh/share/mmsatellite/mtp_bohb/hpbandster_mtp_trials_log.txt'

# Worker class definition: Defines the objective function that BOHB will optimize
class MyMTPWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, config, budget, **kwargs):
        # Set the hyperparameters to optimize
        batch_size = config['batch_size']
        lr = config['lr']
        weight_decay = config['weight_decay']

        # Load the MTP config file
        config_path = '/home/lsh/share/mmsatellite/configs/mtp/rvsa-l-unet-256-mae-mtp_levirtest.py'
        cfg = Config.fromfile(config_path)

        # Apply the optimized hyperparameters to the configuration
        cfg.train_dataloader['batch_size'] = batch_size
        cfg.optim_wrapper['optimizer']['lr'] = lr
        cfg.optim_wrapper['optimizer']['weight_decay'] = weight_decay

        # Save the modified configuration to a temporary path
        modified_config_path = '/home/lsh/share/mmsatellite/mtp_bohb/hpbandster_temp_mtp_config.py'
        cfg.dump(modified_config_path)

        # Log path for individual trial
        log_file_path = f'/home/lsh/share/mmsatellite/mtp_bohb/hpbandster_mtp_trial_log.txt'

        # Run training
        try:
            with open(log_file_path, 'w') as log_file:
                subprocess.run(
                    ['python', '/home/lsh/share/mmsatellite/train.py', '--config', modified_config_path],
                    stdout=log_file, stderr=log_file, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "3"}
                )

            # Extract metrics (mFscore, Precision, Recall) from log
            f1, precision, recall = 0.0, 0.0, 0.0
            with open(log_file_path, 'r') as log_file:
                lines = log_file.readlines()
                for line in lines:
                    if "mFscore" in line:
                        try:
                            parts = line.replace(',', '').split()
                            for i, part in enumerate(parts):
                                if "mFscore" in part:
                                    f1 = float(parts[i + 1])
                                elif "mPrecision" in part:
                                    precision = float(parts[i + 1])
                                elif "mRecall" in part:
                                    recall = float(parts[i + 1])
                        except (IndexError, ValueError) as e:
                            print(f"Error parsing metrics from line: {line} - {e}")

            # Append the results to the unified log file
            with open(all_trials_log_file_path, 'a') as log_file:
                log_file.write(f'Batch size: {batch_size}, LR: {lr}, Weight Decay: {weight_decay}, F1-score: {f1}, Precision: {precision}, Recall: {recall}\n')

        except subprocess.CalledProcessError as e:
            print(f"Training failed: {e}")
            f1, precision, recall = 0.0, 0.0, 0.0

        return ({
            'loss': -f1,  # BOHB minimizes loss, so we return the negative F1 score
            'info': {
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        })

# Define the hyperparameter search space
def get_mtp_configspace():
    config_space = CS.ConfigurationSpace()

    # Batch size: 1 to 4
    batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=1, upper=4, default_value=2)
    config_space.add_hyperparameter(batch_size)

    # Learning rate
    lr = CSH.UniformFloatHyperparameter('lr', lower=6e-5, upper=1e-2, default_value=1e-3, log=True)
    config_space.add_hyperparameter(lr)

    # Weight decay
    weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=0.1, upper=0.5, default_value=0.3)
    config_space.add_hyperparameter(weight_decay)

    return config_space

# BOHB execution setup
result_logger = hpres.json_result_logger(directory='./bohb_mtp_results', overwrite=True)

# Start the NameServer
NS = hpns.NameServer(run_id='bohb_mtp', host='localhost', port=9091)
NS.start()

worker = MyMTPWorker(nameserver='localhost', run_id='bohb_mtp', nameserver_port=9091, timeout=120)
worker.run(background=True)

# Run BOHB optimization
bohb = optimizers.BOHB(
    configspace=get_mtp_configspace(),
    run_id='bohb_mtp',
    nameserver='localhost',
    nameserver_port=9091,
    result_logger=result_logger,
    min_budget=1,   # Minimum budget (epochs)
    max_budget=100  # Maximum budget (epochs)
)

res = bohb.run(n_iterations=50)  # Set the number of iterations to 50

# Shutdown BOHB and NameServer
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# Output the best hyperparameters
best_config = res.get_incumbent_id()
all_runs = res.get_runs_by_id(best_config)
best_run = all_runs[-1]
best_hyperparameters = res.get_id2config_mapping()[best_config]['config']

print(f"Best hyperparameters: {best_hyperparameters}")