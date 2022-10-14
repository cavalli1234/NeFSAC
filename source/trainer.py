import pytorch_lightning as pl
import torch
import numpy as np
from torch import nn
from torch.nn.functional import relu, mish, leaky_relu
from torchmetrics import Accuracy, Precision, Recall, AUROC


class NeFSAC_PL(pl.LightningModule):
    def __init__(self, model, lr=1e-3, train_mode='data'):
        super().__init__()
        self.model = model
        self.train_mode = train_mode
        self.learning_rate = lr
        self.val_acc5 = Accuracy(threshold=0.5)
        self.val_prec0 = Precision(threshold=0.)
        self.val_prec5 = Precision(threshold=0.5)
        self.val_prec75 = Precision(threshold=0.75)
        self.val_prec9 = Precision(threshold=0.9)
        self.val_rec5 = Recall(threshold=0.5)
        self.val_rec75 = Recall(threshold=0.75)
        self.val_auroc = AUROC()
        self.register_buffer('double_y_counts4', torch.ones(size=(4, 2), dtype=torch.long, requires_grad=False))
        self.register_buffer('total_samples4', torch.ones(size=(4, 1), dtype=torch.long, requires_grad=False))

    def error_to_binary_class(self, err, err_min, err_max):
        return (err_max - err.clamp(err_min, err_max)).div(err_max-err_min).clamp(0., 1.)

    def compute_loss(self, y_hat, y, mode):
        counter_idx = {'se': 0, 'rt': 1, 'anrt': 2, 'total': 3}[mode]
        ylong = (y > 0.5).long()
        n_zeros = (1-ylong).sum().item()
        n_ones = ylong.sum().item()
        self.total_samples4[counter_idx] += y.shape[0]
        self.double_y_counts4[counter_idx, 0] += n_zeros * 2
        self.double_y_counts4[counter_idx, 1] += n_ones * 2
        loss = nn.functional.binary_cross_entropy(y_hat, y,
                                                  self.total_samples4[counter_idx].div(self.double_y_counts4[counter_idx][ylong]))
        return loss

    def training_step(self, batch, batch_idx):
        x, rte, an_rte, se = batch
        y = self.error_to_binary_class(rte, 5., 30.) * self.error_to_binary_class(se, 1., 3.)
        if self.train_mode == 'analytical':
            # This training mode trains only on the hand-engineered prior
            y_se = self.error_to_binary_class(se, 2., 5.)
            y_anrt = self.error_to_binary_class(an_rte, 5., 30.)
            y_hat, y_hat_scores = self.model(x, with_partials=True)
            y_rt_hat = y_hat_scores[..., 0]
            loss_anrt = self.compute_loss(y_rt_hat, y_anrt, mode='anrt')
            loss_rt = None
            loss_se = None
            loss_all = loss_anrt
        elif self.train_mode == 'data':
            # This training mode trains only on data, ignoring any expert branch
            # Still, this splits between sampson and pose error (outlier contamination vs degeneracy)
            y_se = self.error_to_binary_class(se, 2., 5.)
            low_se_mask = se <= 2.
            y_rt = self.error_to_binary_class(rte[low_se_mask], 5., 30.)
            y_hat, y_hat_scores = self.model(x, with_partials=True)
            y_se_hat, y_rt_hat = y_hat_scores[..., 0], y_hat_scores[..., 1]
            loss_se = self.compute_loss(y_se_hat, y_se, mode='se')
            loss_rt = self.compute_loss(y_rt_hat[low_se_mask], y_rt, mode='rt')
            loss_anrt = None
            if torch.isnan(loss_rt):
                loss_all = loss_se
            else:
                loss_all = loss_se.add(loss_rt).div(2.)
        elif self.train_mode == 'mixed':
            # This training mode trains all branches, including expert branches, as described in the paper
            y_se = self.error_to_binary_class(se, 2., 5.)
            y_anrt = self.error_to_binary_class(an_rte, 5., 30.)
            low_se_mask = se <= 2.
            y_rt = self.error_to_binary_class(rte[low_se_mask], 5., 30.)
            y_hat, y_hat_scores = self.model(x, with_partials=True)
            y_se_hat, y_rt_hat, y_anrt_hat = y_hat_scores[..., 0], y_hat_scores[..., 1], y_hat_scores[..., 2]
            loss_rt = self.compute_loss(y_rt_hat[low_se_mask], y_rt, mode='rt')
            loss_anrt = self.compute_loss(y_anrt_hat, y_anrt, mode='anrt')
            loss_se = self.compute_loss(y_se_hat, y_se, mode='se')
            if torch.isnan(loss_rt):
                loss_all = loss_anrt.add(loss_se).div(2.)
            else:
                loss_all = loss_rt.add(loss_anrt).add(loss_se).div(3.)
        elif self.train_mode == 'simple':
            # This training mode trains only on pose data, without branching
            y_hat, y_hat_scores = self.model(x, with_partials=True)
            y_all_hat = y_hat_scores[..., 0]
            loss_all = self.compute_loss(y_all_hat, y, mode='rt')
            loss_rt = None
            loss_anrt = None
            loss_se = None
        else:
            raise ValueError(f"Unknown train mode {self.train_mode}")
        loss_all = loss_all + self.compute_loss(y_hat, y, mode='total')
        if loss_rt is not None and not torch.isnan(loss_rt):
            self.log('train/loss_rt', loss_rt, on_step=False, on_epoch=True)
        if loss_se is not None:
            self.log('train/loss_se', loss_se, on_step=False, on_epoch=True)
        if loss_anrt is not None:
            self.log('train/loss_anrt', loss_anrt, on_step=False, on_epoch=True)
        self.log('train/loss_all', loss_all, on_step=False, on_epoch=True)
        return loss_all

    def validation_step(self, batch, batch_idx):
        x, rte, arte, se = batch
        y = self.error_to_binary_class(rte, 5., 30.) * self.error_to_binary_class(se, 1., 3.)
        y_hat = self.model(x)
        ylong = (y > 0.5).long()
        loss = nn.functional.binary_cross_entropy(y_hat, y)
        self.val_acc5.update(y_hat, ylong)
        self.val_prec5.update(y_hat, ylong)
        self.val_rec5.update(y_hat, ylong)
        self.val_prec0.update(y_hat, ylong)
        self.val_prec75.update(y_hat, ylong)
        self.val_prec9.update(y_hat, ylong)
        self.val_rec75.update(y_hat, ylong)
        self.val_auroc.update(y_hat, ylong)
        self.log('val/loss', loss)

    def validation_epoch_end(self, outs):
        self.log('val/acc5',
                 self.val_acc5.compute())
        self.log('val/rec5',
                 self.val_rec5.compute())
        self.log('val/pre0',
                 self.val_prec0.compute())
        self.log('val/pre5',
                 self.val_prec5.compute())
        self.log('val/pre75',
                 self.val_prec75.compute())
        self.log('val/pre9',
                 self.val_prec9.compute())
        self.log('val/rec75',
                 self.val_rec75.compute())
        try:
            auroc = self.val_auroc.compute()
        except ValueError:
            auroc = np.nan
        self.log('val/auroc',
                 auroc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


