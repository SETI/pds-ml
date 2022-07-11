import os
import sys
import math
import numpy as np
import shutil
from pathlib import Path
import splitfolders
from termcolor import colored
from enum import Enum
import copy
import logging
import time

import torch
from torchvision.datasets import ImageFolder

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import TensorBoardLogger

#from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import loggers as pl_loggers
from argparse import ArgumentParser

# Internal Package Imports
from .models import SIMCLR, SIMSIAM, CLASSIFIER, encoders
from utilities import io_utilities as ioUtil

# Dictionary of supported Techniques
supported_techniques = {
    "SIMCLR": SIMCLR.SIMCLR,
    "SIMSIAM": SIMSIAM.SIMSIAM,
    "CLASSIFIER": CLASSIFIER.CLASSIFIER,
}

class SSL:

    # Class Attributes
    
    # The path to store the spite data, models and results
    OUT_PATH = None

    # Pass the data path only if the tool will do it sown splitting
    DATA_PATH = None
    # Also pass the VAL_PATH is you have already split the data
    VAL_PATH = None

    # Set by _init_logging
    log_name = None

    # Set by load_model
    checkpoint_path = None
    encoder = None
    

    def __init__(self,
    DATA_PATH,
    OUT_PATH,
    model,
    technique,
    log_basename,
    VAL_PATH = None,
    val_split = 0.2,
    test_split = 0.0,
    cpus = 1,
    gpus = 1,
    image_size = 256,
    batch_size = 128,
    epochs = 10,
    learning_rate = 1e-3,
    hidden_dim = 128,
    patience = -1,
    ckpt_freq = -1,
    seed = 1729
            ):

        self.DATA_PATH = DATA_PATH 
        self.OUT_PATH = OUT_PATH 
        self.model = model 
        self.technique = technique 
        self.log_basename = log_basename 
        self.VAL_PATH = VAL_PATH 
        self.val_split = val_split 
        self.test_split = test_split 
        self.cpus = cpus 
        self.gpus = gpus 
        self.image_size = image_size 
        self.batch_size = batch_size 
        self.epochs = epochs 
        self.learning_rate = learning_rate 
        self.hidden_dim = hidden_dim 
        self.patience = patience 
        self.ckpt_freq = ckpt_freq 
        self.seed = seed 

        self._init_logging()

    def __repr__(self):
        return ioUtil.print_dictionary(self)

    def _init_logging(self):
        """
        This will set up loggin to print messages to both stdout and a log file
        """

        self.log_name = self.technique + "_" + self.log_basename + ".ckpt"
        file_handler = logging.FileHandler(filename="{}.log".format(self.log_name[:-5]))
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers = [file_handler, stdout_handler]
       
        logging.basicConfig(
            level=logging.INFO, 
            handlers=handlers
        )
        #format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',


    def load_model(self):
        """
        In the method, we first check if the model is a ckpt file. If it is, try loading the checkpoint.
        If the checkpoint doesn't load, we will attempt to get only the encoder to load via the specified technique
        If the model is not a .ckpt file, we will load it as an encoder from our list of supported encoders.
        Finally, if it is none of the above, it could be a user specified .pt file to represent the encoder.
        
        Parameters
        ----------
        
        Returns
        -------
        
        
        """
        
        technique = supported_techniques[self.technique]
        # Add technique specific additional arguments to dictionary passed to model constructor
        parser = ArgumentParser()
        additonal_args, _ = technique.add_model_specific_args(parser).parse_known_args()



        model_options = Enum(
            "Models_Implemented", "resnet18 imagenet_resnet18 resnet50 imagenet_resnet50"
        )
        
        if ".ckpt" in self.model:
            self.checkpoint_path = self.model
        
            try:
                # Add the model specific additional args to the dictionary passed to the technique class constructor
                args = self.__dict__ | additonal_args.__dict__
                return technique.load_from_checkpoint(**args)
            except:
                logging.info("Trying to return model encoder only...")
        
                # there may be a more efficient way to find right technique to load
                for previous_technique in supported_techniques.values():
                    try:
                        self.encoder = previous_technique.load_from_checkpoint(
                            **self.__dict__
                        ).encoder
                        logging.info(
                            colored(
                                f"Successfully found previous model {previous_technique}",
                                "blue",
                            )
                        )
                        break
                    except:
                        continue
        
        # encoder specified
        elif "minicnn" in self.model:
            # special case to make minicnn output variable output embedding size depending on user arg
            output_size = int("".join(x for x in self.model if x.isdigit()))
            self.encoder = encoders.miniCNN(output_size)
            self.encoder.embedding_size = output_size
        elif self.model == model_options.resnet18.name:
            self.encoder = encoders.resnet18(
                pretrained=False,
                first_conv=True,
                maxpool1=True,
                return_all_feature_maps=False,
            )
            self.encoder.embedding_size = 512
        elif self.model == model_options.imagenet_resnet18.name:
            self.encoder = encoders.resnet18(
                pretrained=True,
                first_conv=True,
                maxpool1=True,
                return_all_feature_maps=False,
            )
            self.encoder.embedding_size = 512
        elif self.model == model_options.resnet50.name:
            self.encoder = encoders.resnet50(
                pretrained=False,
                first_conv=True,
                maxpool1=True,
                return_all_feature_maps=False,
            )
            self.encoder.embedding_size = 2048
        elif self.model == model_options.imagenet_resnet50.name:
            self.encoder = encoders.resnet50(
                pretrained=True,
                first_conv=True,
                maxpool1=True,
                return_all_feature_maps=False,
            )
            self.encoder.embedding_size = 2048
        
        # try loading just the encoder
        else:
            logging.info(
                "Trying to initialize just the encoder from a pytorch model file (.pt)"
            )
            try:
                self.encoder = torch.load(self.model)
            except:
                raise Exception("Encoder could not be loaded from path")
            try:
                embedding_size = encoder.embedding_size
            except:
                raise logging.exception(
                    "Your model specified needs to tell me its embedding size. I cannot infer output size yet. Do this by specifying a model.embedding_size in your model instance"
                )
        
        # We are initing from scratch so we need to find out how many classes are in this dataset. This is relevant info for the CLASSIFIER
        self.num_classes = len(ImageFolder(self.DATA_PATH).classes)


       #return technique(**self.__dict__)
        # Add the model specific additional args to the dictionary passed to the technique class constructor
        args = self.__dict__ | additonal_args.__dict__
        return technique(**args)


    def prepare_data(self):
        """
        Prepares the data by splitting the images into a traing, validation and optionally, a test set.
        
        Parameters
        ----------
        
        Returns
        -------
            These attributes change:
            self.DATA_PATH -- now points to the training data subdirectory
            self..VAL_PATH -- now points to the validation data subdirectory
        
        """
        
        logger = logging.getLogger('data_preparation')
        
        # Splitting Data into train and validation
        if (
            not (
                os.path.isdir(f"{self.DATA_PATH}/train")
                and os.path.isdir(f"{self.DATA_PATH}/val")
            )
            and self.val_split != 0
            and self.VAL_PATH is None
        ):
            logger.info("Automatically splitting data into train and validation data...")
            shutil.rmtree(f"{self.OUT_PATH}/split_data_{self.log_name[:-5]}", ignore_errors=True)
            splitfolders.ratio(
                self.DATA_PATH,
                output=f"{self.OUT_PATH}/split_data_{self.log_name[:-5]}",
                ratio=(
                    1 - self.val_split - self.test_split,
                    self.val_split,
                    self.test_split,
                ),
                seed=self.seed,
            )
            self.DATA_PATH = f"{self.OUT_PATH}/split_data_{self.log_name[:-5]}/train"
            self.VAL_PATH = f"{self.OUT_PATH}/split_data_{self.log_name[:-5]}/val"


    def train(self):
        """
        This is the workhorse function that does all the initial SSL training and the fine tuning.
        
        Parameters
        ----------
        
        """
        logger = logging.getLogger('training')
        
        # logging
        #wandb_logger = WandbLogger(name=log_name, project="Curator")
        TBlogger = TensorBoardLogger("tb_logs", name=self.log_name)
        pass

        
        model = self.load_model()
        logger.info("Model architecture successfully loaded")
        
        cbs = []
        backend = "ddp"
        
        if self.patience > 0:
            cb = EarlyStopping("val_loss", patience=self.patience)
            cbs.append(cb)
        ckpt_callback = ModelCheckpoint(
            monitor="train_loss",
            dirpath=os.path.join(os.getcwd(), "models"),
            period=self.ckpt_freq,
            filename="model-{epoch:02d}-{train_loss:.2f}",
        )
        cbs.append(ckpt_callback)
        
        trainer = pl.Trainer(
            gpus=self.gpus,
            max_epochs=self.epochs,
            progress_bar_refresh_rate=20,
            callbacks=cbs,
            distributed_backend=f"{backend}" if self.gpus > 1 else None,
            sync_batchnorm=True if self.gpus > 1 else False,
            logger=TBlogger,
            enable_pl_optimizer=True,
        )
        
        
        startTime = time.time()
        trainer.fit(model)
        endTime = time.time()
        totalTime = endTime - startTime
        logger.info("Total model fitting time: {:.2f} minutes, {:.2f} hours".format(totalTime/60, totalTime/60/60))
        
        Path(f"./models/").mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(f"./models/{self.log_name}")
        logger.info("YOUR MODEL CAN BE ACCESSED AT: ./models/{}".format(self.log_name))


#******************************************************************************************
if __name__ == "__main__":
    """ 
    This is the command line version of the code

    """

    parser = ArgumentParser()
    parser.add_argument(
        "--DATA_PATH", type=str, help="path to folders with images to train on."
    )
    parser.add_argument(
        "--VAL_PATH",
        type=str,
        default=None,
        help="path to validation folders with images",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="model to initialize. Can accept model checkpoint or just encoder name from models.py",
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="batch size for SSL"
    )
    parser.add_argument(
        "--cpus", default=1, type=int, help="number of cpus to use to fetch data"
    )
    parser.add_argument(
        "--gpus", default=1, type=int, help="number of gpus to use for training"
    )
    parser.add_argument(
        "--hidden_dim",
        default=128,
        type=int,
        help="hidden dimensions in projection head or classification layer for finetuning",
    )
    parser.add_argument(
        "--epochs", default=400, type=int, help="number of epochs to train model"
    )
    parser.add_argument(
        "--learning_rate", default=1e-3, type=float, help="learning rate for encoder"
    )
    parser.add_argument(
        "--patience",
        default=-1,
        type=int,
        help="automatically cuts off training if validation does not drop for (patience) epochs. Leave blank to have no validation based early stopping.",
    )
    parser.add_argument(
        "--val_split",
        default=0.2,
        type=float,
        help="percent in validation data. Ignored if VAL_PATH specified",
    )
    parser.add_argument(
        "--test_split",
        default=0,
        type=float,
        help="decimal from 0-1 representing how much of the training data to withold from either training or validation. Used for experimenting with labels neeeded",
    )
    parser.add_argument(
        "--log_name",
        type=str,
        default=None,
        help="name of model to log",
    )
    parser.add_argument(
        "--image_size", default=256, type=int, help="height of square image"
    )
    parser.add_argument(
        "--technique", default=None, type=str, help="SIMCLR, SIMSIAM or CLASSIFIER"
    )
    parser.add_argument(
        "--ckpt_freq", default=-1, type=int, help="Number of epochs between checkpoints"
    )
    parser.add_argument(
        "--seed", default=1729, type=int, help="random seed for run for reproducibility"
    )

    # add ability to parse unknown args
    args, _ = parser.parse_known_args()
    technique = supported_techniques[args.technique]
    args, _ = technique.add_model_specific_args(parser).parse_known_args()


    

