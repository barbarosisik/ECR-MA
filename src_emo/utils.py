from typing import List, Union, Optional
import logging
import os

import torch


def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration for the application."""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def padded_tensor(
    items: List[Union[List[int], torch.LongTensor]],
    pad_idx: int = 0,
    pad_tail: bool = True,
    max_len: Optional[int] = None,
    debug: bool = False,
    device: torch.device = torch.device('cpu'),
    use_amp: bool = False,
    fixed_len = None,
    dtype_=torch.long
) -> torch.LongTensor:
    """Create a padded matrix from an uneven list of lists.

    Returns padded matrix.

    Matrix is right-padded (filled to the right) by default, but can be
    left padded if the flag is set to True.

    Matrix can also be placed on cuda automatically.

    :param list[iter[int]] items: List of items
    :param int pad_idx: the value to use for padding
    :param bool pad_tail:
    :param int max_len: if None, the max length is the maximum item length

    :returns: padded tensor.
    :rtype: Tensor[int64]

    """
    # number of items
    n = len(items)
    # length of each item
    lens: List[int] = [len(item) for item in items]
    # max in time dimension
    t = max(lens)
    # if input tensors are empty, we should expand to nulls
    t = max(t, 1)
    if debug and max_len is not None:
        t = max(t, max_len)
    if fixed_len is not None:
        t = fixed_len

    # if use_amp:
    #     t = t // 8 * 8

    output = torch.full((n, t), fill_value=pad_idx, dtype=dtype_, device=device)

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            continue
        if not isinstance(item, torch.Tensor):
            item = torch.tensor(item, dtype=dtype_, device=device)
        if pad_tail:
            output[i, :length] = item
        else:
            output[i, t - length:] = item

    return output

def padded_matrix(
    items: List[Union[List[int], torch.LongTensor]],
    pad_idx: list,
    pad_tail: bool = True,
    max_len: Optional[int] = None,
    debug: bool = False,
    device: torch.device = torch.device('cpu'),
    use_amp: bool = False
) -> torch.LongTensor:
    """Create a padded matrix from an uneven list of lists.

    Returns padded matrix.

    Matrix is right-padded (filled to the right) by default, but can be
    left padded if the flag is set to True.

    Matrix can also be placed on cuda automatically.

    :param list[iter[int]] items: List of items
    :param int pad_idx: the value to use for padding
    :param bool pad_tail:
    :param int max_len: if None, the max length is the maximum item length

    :returns: padded tensor.
    :rtype: Tensor[int64]

    """
    # number of items
    n = len(items)
    # length of each item
    lens: List[int] = [len(item) for item in items]
    # max in time dimension
    t = max(lens)
    # if input tensors are empty, we should expand to nulls
    t = max(t, 1)
    if debug and max_len is not None:
        t = max(t, max_len)

    if use_amp:
        t = t // 8 * 8

    output = torch.full((n, t), fill_value=pad_idx, dtype=torch.long, device=device)

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            continue
        if not isinstance(item, torch.Tensor):
            item = torch.tensor(item, dtype=torch.long, device=device)
        if pad_tail:
            output[i, :length] = item
        else:
            output[i, t - length:] = item

    return output