import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.utils import WEIGHTS_NAME, logging

logger = logging.get_logger(__name__)


class CLTrainer(Trainer):
    """
    Contrastive Learning trainer that stays close to the stock transformers.Trainer API.
    It keeps compatibility with the existing training scripts while streamlining the
    implementation for modern (CUDA-only) workflows.
    """

    def __init__(self, *args, model_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Persist the SimCSE/BLaIR model args so we can reload checkpoints safely.
        self.model_args = model_args if model_args is not None else getattr(self.model, "model_args", None)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        loss_key = f"{metric_key_prefix}_loss"
        cl_key = f"{metric_key_prefix}_cl_loss"
        if loss_key in metrics:
            metrics[cl_key] = -metrics[loss_key]
            self.log({cl_key: metrics[cl_key]})

        return metrics

    def train(self, *args, **kwargs):
        """
        Overrides the default train loop only to inject a custom best-checkpoint
        loader that works with our contrastive heads. Everything else is delegated
        to transformers.Trainer for maximum compatibility with new releases.
        """
        should_reload_best = bool(self.args.load_best_model_at_end)
        if should_reload_best:
            logger.info(
                "load_best_model_at_end=True detected. Falling back to CLTrainer's "
                "state-dict loader to stay compatible with custom heads."
            )
            self.args.load_best_model_at_end = False

        try:
            train_output = super().train(*args, **kwargs)
        finally:
            if should_reload_best:
                self.args.load_best_model_at_end = True

        if should_reload_best:
            self._load_best_checkpoint_weights()

        return train_output

    def _load_best_checkpoint_weights(self) -> None:
        """
        Reloads the best checkpoint by copying its state dict into the current model.
        This keeps multi-GPU/AMP setups happy and avoids re-instantiating the class
        (which would otherwise require passing the custom model_args through
        transformers' from_pretrained stack).
        """
        checkpoint_dir = self.state.best_model_checkpoint
        if not checkpoint_dir:
            logger.warning(
                "load_best_model_at_end was enabled but no best checkpoint was recorded. "
                "Was metric_for_best_model set correctly?"
            )
            return

        weights_path = os.path.join(checkpoint_dir, WEIGHTS_NAME)
        if not os.path.exists(weights_path):
            logger.warning("Best checkpoint detected at %s but %s is missing.", checkpoint_dir, WEIGHTS_NAME)
            return

        if self.is_world_process_zero():
            logger.info(
                "Loading best checkpoint from %s (metric: %s).",
                checkpoint_dir,
                self.state.best_metric,
            )

        state_dict = torch.load(weights_path, map_location="cpu")
        target_model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
        target_model.load_state_dict(state_dict, strict=True)

        # Keep the wrapped copy in sync for any downstream calls.
        if hasattr(self, "model_wrapped"):
            self.model_wrapped = self.model
    