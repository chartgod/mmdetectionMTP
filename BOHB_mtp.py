import os
import subprocess
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import hpbandster.optimizers as optimizers
from hpbandster.core.worker import Worker

# mmengine 또는 mmcv에서 Config 가져오기
from mmengine.config import Config  # 또는 mmcv.Config

# 로그 파일 경로 설정
all_trials_log_file_path = '/home/lsh/share/mmsatellite/mtp_bohb/mtp_hpbandster_trials_log34.txt'

# Worker 클래스 정의: BOHB가 최적화할 목표 함수를 정의
class MyWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, config, budget, **kwargs):
        # 하이퍼파라미터를 설정
        batch_size = config['batch_size']
        lr = config['lr']
        weight_decay = config['weight_decay']

        # 모델 설정 파일의 경로를 변경하여 최적화된 값으로 업데이트
        config_path = '/home/lsh/share/mmsatellite/configs/mtp/rvsa-l-unet-256-mae-mtp_levir.py'
        cfg = Config.fromfile(config_path)

        # 설정 파일에 하이퍼파라미터 적용
        cfg.train_dataloader['batch_size'] = batch_size
        cfg.optimizer['lr'] = lr
        cfg.optimizer['weight_decay'] = weight_decay

        # 임시 경로에 수정된 config 파일 저장
        modified_config_path = '/home/lsh/share/mmsatellite/mtp_bohb/mtp_hpbandster_temp_config.py'
        cfg.dump(modified_config_path)

        # 학습 로그를 파일로 저장하기 위해 로그 경로 지정
        log_file_path = f'/home/lsh/share/mmsatellite/mtp_bohb/mtp_hpbandster_trial_log34.txt'

        # 학습 시작
        try:
            with open(log_file_path, 'w') as log_file:
                # GPU 1, 2, 3, 4, 5, 6, 7번에 할당되도록 환경 변수 설정
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"  # GPU 1~7번 사용

                # 학습 프로세스 실행 (train_multi1234567.py 실행)
                subprocess.run(
                    ['python', '/home/lsh/share/mmsatellite/train_multi1234567.py', '--config', modified_config_path],
                    stdout=log_file, stderr=log_file, check=True, env=env  # 환경 변수 전달
                )

            # 성능 메트릭(mFscore, Precision, Recall) 로그 파일에서 읽기
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

            # 통합된 로그 파일에 기록 (사용된 하이퍼파라미터와 함께 기록)
            with open(all_trials_log_file_path, 'a') as log_file:
                log_file.write(f'Batch size: {batch_size}, LR: {lr}, Weight Decay: {weight_decay}, F1-score: {f1}, Precision: {precision}, Recall: {recall}\n')

        except subprocess.CalledProcessError:
            f1, precision, recall = 0.0, 0.0, 0.0

        return ({
            'loss': -f1,  # BOHB는 최소화 문제로 설정되어 있으므로 음수 값을 반환
            'info': {
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        })


# ConfigSpace 생성: 하이퍼파라미터의 탐색 공간 정의
def get_configspace():
    config_space = CS.ConfigurationSpace()

    # Batch size
    batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=1, upper=32, default_value=16)
    config_space.add_hyperparameter(batch_size)

    # Learning rate
    lr = CSH.UniformFloatHyperparameter('lr', lower=6e-5, upper=1e-2, default_value=1e-3, log=True)
    config_space.add_hyperparameter(lr)

    # Weight decay
    weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=0.1, upper=0.5, default_value=0.3)
    config_space.add_hyperparameter(weight_decay)

    return config_space


# BOHB 실행 설정
result_logger = hpres.json_result_logger(directory='./bohb_results', overwrite=True)

# NameServer 실행, 포트를 명확하게 지정
NS = hpns.NameServer(run_id='bohb', host='localhost', port=9093)
NS.start()

worker = MyWorker(nameserver='localhost', run_id='bohb', nameserver_port=9093, timeout=120)
worker.run(background=True)

# BOHB 최적화 실행
bohb = optimizers.BOHB(
    configspace=get_configspace(),
    run_id='bohb',
    nameserver='localhost',
    nameserver_port=9093,
    result_logger=result_logger,
    min_budget=1,   # 최소 budget (epoch 수)
    max_budget=100  # 최대 budget (epoch 수)
)

res = bohb.run(n_iterations=100)  # 최적화 반복 횟수

# 최종 결과 확인
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# 결과에서 최적의 하이퍼파라미터 출력
best_config = res.get_incumbent_id()
all_runs = res.get_runs_by_id(best_config)
best_run = all_runs[-1]  # 최종 실행된 트라이얼에서 가장 좋은 결과
best_hyperparameters = res.get_id2config_mapping()[best_config]['config']

print(f"Best hyperparameters: {best_hyperparameters}")
