import sys
import subprocess
def install(package):
    subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package])
    
install('dvc==2.8.3')
install('s3fs==2021.11.0')
install('dvc[s3]==2.8.3')
install('git-remote-codecommit')
install('sagemaker-experiments')
install('gitpython')
install('scikit-learn')
install('pytorch-lightning')
install('timm')
install('tensorboard')
install('tensorboardX')

from typing import Any, Dict, Optional, Tuple

import os
import subprocess
import torch
import timm
import json
import tarfile

import pytorch_lightning as pl
import torchvision.transforms as T
import torch.nn.functional as F

from pathlib import Path
from torchvision.datasets import ImageFolder
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy
from pytorch_lightning import loggers as pl_loggers
from datetime import datetime

from train import LitResnet
from train import IntelImgClfDataModule

ml_root = Path("/opt/ml")

model_artifacts = ml_root / "processing" / "model"
dataset_dir = ml_root / "processing"

def eval_model(trainer, model, datamodule):
    test_res = trainer.test(model, datamodule)[0]
    
    report_dict = {
        "multiclass_classification_metrics": {
            "accuracy": {
                "value": test_res["test/acc"],
                "standard_deviation": "0",
            },
        },
    }
    
    eval_folder = ml_root / "processing" / "evaluation"
    eval_folder.mkdir(parents=True, exist_ok=True)
    
    out_path = eval_folder / "evaluation.json"
    
    print(f":: Writing to {out_path.absolute()}")
    
    with out_path.open("w") as f:
        f.write(json.dumps(report_dict))
        
        
if __name__ == '__main__':
    
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    datamodule = FlowerDataModule(
        data_dir=dataset_dir.absolute(),
        num_workers=os.cpu_count()
    )
    datamodule.setup()
    
    model = LitResnet.load_from_checkpoint(checkpoint_path="last.ckpt")
    
    trainer = pl.Trainer(
        accelerator="auto",
    )
    
    print(":: Evaluating Model")
    eval_model(trainer, model, datamodule)