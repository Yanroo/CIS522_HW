from typing import List
import warnings
from torch.optim.lr_scheduler import _LRScheduler
from config import CONFIG


class CustomLRScheduler(_LRScheduler):
    """Custom learning rate scheduler."""

    def __init__(self, optimizer, gamma=0.1, step_size=1, last_epoch=-1, verbose=False):
        """
        Decays the learning rate of each parameter group by gamma every
        step_size epochs. Notice that such decay can happen simultaneously with
        other changes to the learning rate from outside this scheduler. When
        last_epoch=-1, sets initial lr as lr.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            step_size (int): Period of learning rate decay.
            gamma (float): Multiplicative factor of learning rate decay. Default: 0.1.
            last_epoch (int): The index of last epoch. Default: -1.
            verbose (bool): If ``True``, prints a message to stdout for
                each update. Default: ``False``.

        """
        # ... Your Code Here ...
        self.gamma = gamma
        self.step_size = step_size
        self.warmup_steps = 3 * 50000 // CONFIG.batch_size
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """Return the current learning rate."""
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )
        # if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
        #     lr = [group['lr'] for group in self.optimizer.param_groups]
        #     return lr
        # return [group['lr'] * self.gamma
        #         for group in self.optimizer.param_groups]
        step = self._step_count
        return [0.1 * min(step**-0.5, step * self.warmup_steps**-1.5)]
        # return [-2.28e-7 * step + 3e-3]

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """Display the current learning rate."""
        # super(CustomLRScheduler, self).print_lr(is_verbose, group, lr, epoch)
        steps_per_epoch = 50000 // CONFIG.batch_size
        if is_verbose:
            if self._step_count % steps_per_epoch == 0:
                print(
                    "Adjusting learning rate"
                    " of group {} to {:.4e}.".format(group, lr)
                )

    # def step(self, epoch=None):
    #     super().step(epoch)
    #     if epoch is not None:
    #         print("Learning rate:", self.get_lr())
    # ... Your Code Here ...

    # Here's our dumb baseline implementation:
    # return [i for i in self.base_lrs]
