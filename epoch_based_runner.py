# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import time
import warnings
import torch
import mmcv
from .base_runner import BaseRunner
from .checkpoint import save_checkpoint
from .utils import get_host_info


class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def train(self, data_loader, **kwargs):
        if type(data_loader) != list:
            self.model.train()
            self.mode = "train"
            self.data_loader = data_loader
            self._max_iters = self._max_epochs * len(data_loader)
            self.call_hook("before_train_epoch")
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            for i, X_L in enumerate(data_loader):
                X_L.update({"x": X_L.pop("img")})
                X_L.update({"y_loc_img": X_L.pop("gt_bboxes")})
                X_L.update({"y_cls_img": X_L.pop("gt_labels")})
                self._inner_iter = i
                self.call_hook("before_train_iter")
                if self.batch_processor is None:
                    outputs = self.model.train_step(X_L, self.optimizer, **kwargs)
                else:
                    outputs = self.batch_processor(self.model, X_L, train_mode=True, **kwargs)
                if not isinstance(outputs, dict):
                    raise TypeError('"batch_processor()" or "model.train_step()" must return a dict')
                if "log_vars" in outputs:
                    self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
                self.outputs = outputs
                self.call_hook("after_train_iter")
                self._iter += 1
            self.call_hook("after_train_epoch")
            self._epoch += 1
        else:
            self.model.train()
            self.mode = "train"
            self.data_loader = data_loader[0]
            self._max_iters = self._max_epochs * len(data_loader[0])
            self.call_hook("before_train_epoch")
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            unlabeled_data_iter = iter(data_loader[1])
            for i, X_L in enumerate(data_loader[0]):
                X_L.update({"x": X_L.pop("img")})
                X_L.update({"y_loc_img": X_L.pop("gt_bboxes")})
                X_L.update({"y_cls_img": X_L.pop("gt_labels")})
                self._inner_iter = i
                self.call_hook("before_train_iter")
                if self.batch_processor is None:
                    outputs = self.model.train_step(X_L, self.optimizer, **kwargs)
                else:
                    outputs = self.batch_processor(self.model, X_L, train_mode=True, **kwargs)
                    error("Not Implemeted!")
                if not isinstance(outputs, dict):
                    raise TypeError('"batch_processor()" or "model.train_step()" must return a dict')
                if "log_vars" in outputs:
                    self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
                self.outputs = outputs
                self.call_hook("after_train_iter")
                X_U = unlabeled_data_iter.next()
                X_U.update({"x": X_U.pop("img")})
                X_U.update({"y_loc_img": X_U.pop("gt_bboxes")})
                X_U.update({"y_cls_img": X_U.pop("gt_labels")})
                X_U = self.clear_gt_label(X_U)
                self._inner_iter = i
                self.call_hook("before_train_iter")
                if self.batch_processor is None:
                    outputs = self.model.train_step(X_U, self.optimizer, **kwargs)
                else:
                    outputs = self.batch_processor(self.model, X_U, train_mode=True, **kwargs)
                    error("Not Implemeted!")
                if not isinstance(outputs, dict):
                    raise TypeError('"batch_processor()" or "model.train_step()" must return a dict')
                if "log_vars" in outputs:
                    self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
                self.outputs = outputs
                self.call_hook("after_train_iter")
                self._iter += 1
            self.call_hook("after_train_epoch")
            self._epoch += 1

    def clear_gt_label(self, X_U):
        BatchSize = len(X_U["y_cls_img"].data[0])
        for i in range(BatchSize):
            X_U["y_loc_img"].data[0][i].fill_(-1)
        return X_U

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = "val"
        self.data_loader = data_loader
        self.call_hook("before_val_epoch")
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, X_val in enumerate(data_loader):
            X_L.update({"x": X_L.pop("img")})
            X_val.update({"y_loc_img": X_val.pop("gt_bboxes")})
            X_val.update({"y_cls_img": X_val.pop("gt_labels")})
            self._inner_iter = i
            self.call_hook("before_val_iter")
            with torch.no_grad():
                if self.batch_processor is None:
                    outputs = self.model.val_step(X_val, self.optimizer, **kwargs)
                else:
                    outputs = self.batch_processor(self.model, X_val, train_mode=False, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('"batch_processor()" or "model.val_step()" must return a dict')
            if "log_vars" in outputs:
                self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
            self.outputs = outputs
            self.call_hook("after_val_iter")
        self.call_hook("after_val_epoch")

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        if type(data_loaders[0]) != list:
            assert isinstance(data_loaders, list)
            assert mmcv.is_list_of(workflow, tuple)
            assert len(data_loaders) == len(workflow)
            self._max_epochs = max_epochs
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if mode == "train":
                    self._max_iters = self._max_epochs * len(data_loaders[i])
                    break
            work_directory = self.work_dir if self.work_dir is not None else "NONE"
            self.logger.info("Start running, host: %s, work_directory: %s", get_host_info(), work_directory)
            self.logger.info("workflow: %s, max: %d epochs", workflow, max_epochs)
            self.call_hook("before_run")
            while self.epoch < max_epochs:
                for i, flow in enumerate(workflow):
                    mode, epochs = flow
                    if isinstance(mode, str):  # self.train()
                        if not hasattr(self, mode):
                            raise ValueError(f'runner has no method named "{mode}" to run an epoch')
                        epoch_runner = getattr(self, mode)
                    else:
                        raise TypeError("mode in workflow must be a str, but got {}".format(type(mode)))
                    for _ in range(epochs):
                        if mode == "train" and self.epoch >= max_epochs:
                            break
                        epoch_runner(data_loaders[i], **kwargs)
            time.sleep(1)  # wait for some hooks like loggers to finish
            self.call_hook("after_run")
        else:
            data_loaders_u = data_loaders[1]
            data_loaders = data_loaders[0]
            assert isinstance(data_loaders, list)
            assert mmcv.is_list_of(workflow, tuple)
            assert len(data_loaders) == len(workflow)
            self._max_epochs = max_epochs
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if mode == "train":
                    self._max_iters = self._max_epochs * len(data_loaders[i])
                    break
            work_directory = self.work_dir if self.work_dir is not None else "NONE"
            self.logger.info("Start running, host: %s, work_directory: %s", get_host_info(), work_directory)
            self.logger.info("workflow: %s, max: %d epochs", workflow, max_epochs)
            self.call_hook("before_run")
            while self.epoch < max_epochs:
                for i, flow in enumerate(workflow):
                    mode, epochs = flow
                    if isinstance(mode, str):  # self.train()
                        if not hasattr(self, mode):
                            raise ValueError(f'runner has no method named "{mode}" to run an epoch')
                        epoch_runner = getattr(self, mode)
                    else:
                        raise TypeError("mode in workflow must be a str, but got {}".format(type(mode)))
                    for _ in range(epochs):
                        if mode == "train" and self.epoch >= max_epochs:
                            break
                        epoch_runner([data_loaders[i], data_loaders_u[i]], **kwargs)
            time.sleep(1)  # wait for some hooks like loggers to finish
            self.call_hook("after_run")

    def save_checkpoint(
        self, out_dir, filename_tmpl="epoch_{}.pth", save_optimizer=True, meta=None, create_symlink=True
    ):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(f"meta should be a dict or None, but got {type(meta)}")
        if self.meta is not None:
            meta.update(self.meta)
        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            mmcv.symlink(filename, osp.join(out_dir, "latest.pth"))


class Runner(EpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn("Runner was deprecated, please use EpochBasedRunner instead")
        super().__init__(*args, **kwargs)
