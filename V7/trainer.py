import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import time


class KTEvaluator:
    """KT评估器：计算AUC和Accuracy"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_labels = []

    def update(self, preds, labels):
        """
        Args:
            preds: (batch,) 预测概率
            labels: (batch,) 真实标签 (0/1)
        """
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())

    def compute(self):
        """计算AUC和Accuracy"""
        if len(self.all_preds) == 0:
            return {'auc': 0.0, 'acc': 0.0}

        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        # AUC
        try:
            auc = roc_auc_score(labels, preds)
        except ValueError:
            # 如果只有一个类别，AUC无法计算
            auc = 0.0

        # Accuracy (阈值0.5)
        pred_binary = (preds >= 0.5).astype(int)
        acc = accuracy_score(labels, pred_binary)

        return {'auc': auc, 'acc': acc}


class KTTrainer:
    """KT模型训练器"""

    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        test_loader=None,
        lr=0.001,
        weight_decay=0.0,
        device='cuda',
        patience=10,
        save_path='best_model.pt',
        relation_matrix=None,
        checkpoint_metadata=None,
        sequence_mode=False,
        prior_loss_weight=0.0,
        long_loss_weight=0.0,
        short_loss_weight=0.0,
        mastery_loss_weight=0.0,
        validation_auxiliary_loss=True,
        scheduler_type='step',
        scheduler_step_size=50,
        scheduler_gamma=0.5,
        early_stop_metric='loss',
        grad_clip_norm=1.0,
        progress_path=None,
        progress_metadata=None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device
        self.patience = patience
        self.save_path = save_path
        self.relation_matrix = relation_matrix
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.sequence_mode = sequence_mode
        self.prior_loss_weight = float(prior_loss_weight)
        self.long_loss_weight = float(long_loss_weight)
        self.short_loss_weight = float(short_loss_weight)
        self.mastery_loss_weight = float(mastery_loss_weight)
        self.validation_auxiliary_loss = bool(validation_auxiliary_loss)
        self.scheduler_type = str(scheduler_type).lower()
        self.early_stop_metric = str(early_stop_metric).lower()
        self.grad_clip_norm = float(grad_clip_norm)
        self.progress_path = (
            Path(progress_path).resolve() if progress_path is not None else None
        )
        self.progress_metadata = dict(progress_metadata or {})
        if self.early_stop_metric not in {'loss', 'auc'}:
            raise ValueError("early_stop_metric must be 'loss' or 'auc'")

        # 损失函数和优化器
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # 学习率调度器
        if self.scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(scheduler_step_size),
                gamma=float(scheduler_gamma),
            )
        elif self.scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
            )
        elif self.scheduler_type == 'none':
            self.scheduler = None
        else:
            raise ValueError("scheduler_type must be 'step', 'plateau', or 'none'")

        # 评估器
        self.evaluator = KTEvaluator()

        # 训练历史
        self.history = {
            'train_loss': [],
            'valid_auc': [],
            'valid_acc': [],
            'valid_loss': [],
            'test_auc': [],
            'test_acc': [],
            'best_epoch': None,
            'epoch_duration_seconds': [],
            'learning_rate': [],
            'train_batch_loss': [],
        }

        self.best_valid_auc = -float('inf')
        self.best_selection_value = (
            float('inf') if self.early_stop_metric == 'loss' else -float('inf')
        )
        self.early_stop_counter = 0

    def persist_progress(self, status='running', elapsed_seconds=None):
        """Atomically retain raw epoch/batch history without marking a fold done."""
        if self.progress_path is None:
            return
        payload = {
            'schema_version': 1,
            'status': str(status),
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            'completed_epochs': len(self.history['train_loss']),
            'elapsed_seconds': (
                None if elapsed_seconds is None else float(elapsed_seconds)
            ),
            'checkpoint_path': str(self.save_path),
            'best_valid_auc': float(self.best_valid_auc),
            'best_selection_value': float(self.best_selection_value),
            'early_stop_counter': int(self.early_stop_counter),
            'history': self.history,
            **self.progress_metadata,
        }
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.progress_path.with_suffix(
            self.progress_path.suffix + '.tmp'
        )
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        temporary.replace(self.progress_path)

    def _loss_from_outputs(self, outputs, labels, mask=None, include_auxiliary=True):
        prediction_logits = outputs['logits']
        logits = (
            outputs.get('main_logits', prediction_logits)
            if include_auxiliary else prediction_logits
        )
        if mask is not None:
            logits = logits[mask]
            prediction_logits = prediction_logits[mask]
            labels = labels[mask]
        if labels.numel() == 0:
            return outputs['logits'].sum() * 0.0, logits, labels
        loss = self.criterion(logits, labels)
        for key, weight in [
            ('prior_logits', self.prior_loss_weight),
            ('long_logits', self.long_loss_weight),
            ('short_logits', self.short_loss_weight),
            ('mastery_logits', self.mastery_loss_weight),
        ]:
            if not include_auxiliary or weight <= 0.0:
                continue
            if key not in outputs:
                raise ValueError(
                    f'{key} is required when its auxiliary loss weight is positive'
                )
            auxiliary_logits = outputs[key]
            if mask is not None:
                auxiliary_logits = auxiliary_logits[mask]
            loss = loss + weight * self.criterion(auxiliary_logits, labels)
        return loss, torch.sigmoid(prediction_logits), labels

    def _sequence_loss(self, batch, include_auxiliary=True):
        outputs = self.model.forward_sequence(
            batch, self.relation_matrix, return_aux=True
        )
        labels = batch['response_seq'][:, 1:]
        mask = batch['predict_mask']
        return self._loss_from_outputs(
            outputs, labels, mask=mask, include_auxiliary=include_auxiliary
        )

    def _prefix_loss(self, batch, include_auxiliary=True):
        outputs = self.model(batch, self.relation_matrix, return_aux=True)
        return self._loss_from_outputs(
            outputs,
            batch['target_response'],
            include_auxiliary=include_auxiliary,
        )

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        batch_losses = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_index, batch in enumerate(pbar, start=1):
            # 移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 前向传播
            self.optimizer.zero_grad()
            if self.sequence_mode:
                loss, preds, _ = self._sequence_loss(batch)
                if preds.numel() == 0:
                    continue
            else:
                loss, preds, _ = self._prefix_loss(batch)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f'non-finite training loss at epoch={epoch}, '
                    f'batch={batch_index}'
                )

            # 反向传播
            loss.backward()

            # 梯度裁剪
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip_norm,
                    error_if_nonfinite=True,
                )
            except RuntimeError as error:
                invalid = [
                    name for name, parameter in self.model.named_parameters()
                    if parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ]
                names = ', '.join(invalid[:12]) or 'unknown'
                raise FloatingPointError(
                    f'non-finite gradient norm at epoch={epoch}, '
                    f'batch={batch_index}; parameters={names}'
                ) from error

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            batch_losses.append(float(loss.item()))

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'grad': f'{float(grad_norm):.3f}',
            })

        avg_loss = total_loss / n_batches
        return avg_loss, batch_losses

    @torch.no_grad()
    def evaluate(self, loader, desc='Eval'):
        """评估模型"""
        self.model.eval()
        self.evaluator.reset()

        pbar = tqdm(loader, desc=desc)
        total_loss = 0.0
        n_batches = 0
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 前向传播
            if self.sequence_mode:
                loss, preds, labels = self._sequence_loss(
                    batch, include_auxiliary=self.validation_auxiliary_loss
                )
            else:
                loss, preds, labels = self._prefix_loss(
                    batch, include_auxiliary=self.validation_auxiliary_loss
                )
            if labels.numel() == 0:
                continue
            total_loss += loss.item()
            n_batches += 1

            # 更新评估器
            self.evaluator.update(preds, labels)

        # 计算指标
        metrics = self.evaluator.compute()
        metrics['loss'] = total_loss / max(n_batches, 1)
        return metrics

    def fit(self, epochs):
        """训练模型"""
        print(f"Start training for {epochs} epochs...")
        start_time = time.time()
        self.persist_progress(status='running', elapsed_seconds=0.0)

        for epoch in range(1, epochs + 1):
            # 训练
            epoch_start = time.time()
            train_loss, batch_losses = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_batch_loss'].append(batch_losses)

            # 验证
            valid_metrics = self.evaluate(self.valid_loader, desc=f'Epoch {epoch} [Valid]')
            self.history['valid_auc'].append(valid_metrics['auc'])
            self.history['valid_acc'].append(valid_metrics['acc'])
            self.history['valid_loss'].append(valid_metrics['loss'])

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                  f"valid_loss={valid_metrics['loss']:.4f}, "
                  f"valid_auc={valid_metrics['auc']:.4f}, "
                  f"valid_acc={valid_metrics['acc']:.4f}")

            # 学习率调度
            if self.scheduler_type == 'plateau':
                self.scheduler.step(valid_metrics['auc'])
            elif self.scheduler is not None:
                self.scheduler.step()
            self.history['learning_rate'].append(
                float(self.optimizer.param_groups[0]['lr'])
            )

            # 保存最优模型
            self.best_valid_auc = max(self.best_valid_auc, valid_metrics['auc'])
            selection_value = valid_metrics[self.early_stop_metric]
            improved = (
                selection_value < self.best_selection_value
                if self.early_stop_metric == 'loss'
                else selection_value > self.best_selection_value
            )
            if improved:
                self.best_selection_value = selection_value
                self.early_stop_counter = 0
                self.history['best_epoch'] = epoch
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'best_epoch': epoch,
                    'selection_metric': self.early_stop_metric,
                    'selection_value': selection_value,
                    **self.checkpoint_metadata,
                }
                torch.save(checkpoint, self.save_path)
                print(
                    f"  Best model saved ({self.early_stop_metric}="
                    f"{selection_value:.4f})"
                )
            else:
                self.early_stop_counter += 1

            self.history['epoch_duration_seconds'].append(
                float(time.time() - epoch_start)
            )

            # 早停
            if self.early_stop_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                self.persist_progress(
                    status='early_stopped', elapsed_seconds=time.time() - start_time
                )
                break

            self.persist_progress(
                status='running', elapsed_seconds=time.time() - start_time
            )

            print()

        total_time = time.time() - start_time
        print(f"Training completed in {total_time/60:.2f} minutes")

        # 加载最优模型
        # Sparse graph buffers can fail validation when deserialized directly
        # onto CUDA on some PyTorch builds. Restore on CPU, then let
        # load_state_dict copy tensors into the already-placed model.
        checkpoint = torch.load(
            self.save_path, map_location='cpu', weights_only=False
        )
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        print(
            f"Loaded epoch {self.history['best_epoch']} selected by "
            f"validation {self.early_stop_metric}={self.best_selection_value:.4f}"
        )

        # 测试
        if self.test_loader is not None:
            print("\nEvaluating on test set...")
            test_metrics = self.evaluate(self.test_loader, desc='Test')
            print(f"Test AUC: {test_metrics['auc']:.4f}, "
                  f"Test Accuracy: {test_metrics['acc']:.4f}")

            self.history['test_auc'].append(test_metrics['auc'])
            self.history['test_acc'].append(test_metrics['acc'])

        self.persist_progress(
            status='training_completed', elapsed_seconds=total_time
        )

        return self.history

    def predict(self, loader):
        """预测"""
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(loader, desc='Predicting'):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                preds = self.model(batch, self.relation_matrix)
                all_preds.extend(preds.cpu().numpy())

        return np.array(all_preds)
