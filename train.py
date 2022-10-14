import torch
import pytorch_lightning as pl
from source.trainer import NeFSAC_PL
from source.model import NeFSAC_score_model
from source.data import KittiSequenceDataset


def main(checkpoint_in, checkpoint_out, tensorboard_path, kitti_base_path, data_cache_path=None, sample_size=5, use_gpu=True):
    """
    Reproduce the training setup for NeFSAC on KITTI
    """
    train_dataset = KittiSequenceDataset([0, 1, 2, 3, 4],
                                         bsize=1024,
                                         interval=[1, 2, 3, 4, 5, 6, 7],
                                         sample_size=sample_size,
                                         maxlen=2,
                                         kitti_base_path=kitti_base_path,
                                         data_cache_path=data_cache_path)
    val_dataset = KittiSequenceDataset(5,
                                       bsize=1024,
                                       interval=[1, 2, 3, 4, 5, 6, 7],
                                       sample_size=sample_size,
                                       maxlen=2,
                                       kitti_base_path=kitti_base_path,
                                       data_cache_path=data_cache_path)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=None)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=None)
    callbacks = [pl.callbacks.progress.ProgressBar(refresh_rate=25)]
    if checkpoint_out is not None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(checkpoint_out, monitor='val/loss', mode='min', save_top_k=1)
        callbacks.append(checkpoint_callback)
    if tensorboard_path is not None:
        logger = pl.loggers.TensorBoardLogger(tensorboard_path)
    else:
        logger = None
    trainer = pl.Trainer(max_epochs=10000, gpus=1 if use_gpu else 0, callbacks=callbacks, log_every_n_steps=20, logger=logger)
    pl_model = NeFSAC_PL(NeFSAC_score_model(sample_size=sample_size, branches_out=3), lr=1e-3, train_mode='mixed')
    if checkpoint_in is not None:
        sd = torch.load(checkpoint_in)
        pl_model.load_state_dict(sd['state_dict'], strict=False)
    trainer.fit(pl_model, train_dl, val_dl)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_path", type=str,
                    help="Root path of the KITTI dataset. This directory should contain the two sub-directories 'poses' and 'sequences'.", required=True)
    ap.add_argument("--checkpoint_in", type=str,
                    help="Checkpoint path to resume the training from. No checkpoint by default", default=None)
    ap.add_argument("--checkpoint_out", type=str,
                    help="Checkpoint path to save the training. No checkpointing by default", default=None)
    ap.add_argument("--data_cache_path", type=str,
                    help="Cache preprocessed data to speed up training over time. No caching by default.", default=None)
    ap.add_argument("--tensorboard_path", type=str,
                    help="Path to save tensorboard logs. No logging by default", default=None)
    ap.add_argument("--fundamental", action='store_true',
                    help="Switch to the fundamental matrix estimation task. By default, train on essential matrix estimation.")
    ap.add_argument("--nocuda", action='store_true',
                    help="Switch to training on CPU only. Use one GPU by default.")

    args = ap.parse_args()

    main(checkpoint_in=args.checkpoint_in,
         checkpoint_out=args.checkpoint_out,
         tensorboard_path=args.tensorboard_path,
         data_cache_path=args.data_cache_path,
         kitti_base_path=args.kitti_path,
         sample_size=7 if args.fundamental else 5,
         use_gpu=not args.nocuda)


