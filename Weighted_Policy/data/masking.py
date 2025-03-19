import torch
import numpy as np
from torch import nn
import contextlib

from typing import Optional, Sequence, Tuple, Union

@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[Union[int, Tuple[int, ...]]] = None):
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class AllStepMaskFunc:
    """

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.

    When called, ``AllStepMaskFunc`` uses internal functions create mask by
    1) creating a mask for the k-space center,
    2) create a mask outside of the k-space center, and
    3) combining them into a total mask.
    The internals are handled by ``sample_mask``, which calls ``calculate_center_mask``
    for (1) and ``calculate_acceleration_mask`` for (2).
    The combination is executed in the ``MaskFunc`` ``__call__`` function.
    """

    def __init__(self, center_fractions: Sequence[float], acquisition_range: Sequence[int],
                 allow_any_combination: bool = True, seed: Optional[int] = None):
        """

        Args:
            center_fractions (Sequence[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly each time.
            accelerations (Sequence[int]): Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided, then one of these is chosen uniformly each time.
            allow_any_combination (bool, optional): Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``. Defaults to True.
            seed (Optional[int], optional): Seed for starting the internal random number generator of the
                ``MaskFunc``. Defaults to None.
        """

        # assert len(center_fractions) == len(
        #     accelerations), 'Number of center fractions should match the number of accelerations.'

        self.center_fractions = center_fractions
        self.acquisition_range = acquisition_range
        self.allow_any_combination = allow_any_combination
        self.rng = np.random.RandomState(seed)

    def __call__(self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None) -> Tuple[torch.Tensor, int]:
        """
        Sample and return a k-space mask

        Args:
            shape (Sequence[int]): Shape of the mask to be created.
            seed (Optional[Union[int, Tuple[int, ...]]], optional): seed for RNG for reproducibility . Defaults to None.

        Returns:
            Tuple[torch.Tensor, int]: A 2-tuple containing
            1) the k-space mask and
            2) the number of center frequency lines
        """

        assert len(shape) >= 3, "Shape should have 3 or more dimensions"

        with temp_seed(self.rng, seed):
            center_mask, accel_mask, num_low_frequencies = self.sample_mask(shape)
        return torch.max(center_mask, accel_mask), num_low_frequencies

    def choose_acquisition_step(self, num_cols):
        start_step = num_cols // self.acquisition_range[0]
        end_step = num_cols // self.acquisition_range[1]
        acquisition_step_range = [start_step + i for i in range(end_step - start_step + 1)]
        # random combination
        if self.allow_any_combination:
            return self.rng.choice(self.center_fractions), self.rng.choice(acquisition_step_range)
        # choose according to the same index
        else:
            choice = self.rng.randint(len(self.center_fractions))
            return self.center_fractions[choice], self.accelerations[choice]

    def calculate_center_mask(self, shape: Sequence[int], num_low_freqs: int):
        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs) // 2
        mask[pad: pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs, "Center fractions should be equal to the number of low frequencies"

        return mask

    def calculate_acceleration_mask(self, num_cols, acquis_step, num_low_freqs):
        prob = (acquis_step - num_low_freqs) / (num_cols - num_low_freqs)

        return self.rng.uniform(size=num_cols) < prob

    def reshape_mask(self, mask, shape):
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols  # [1, width, 1]

        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def sample_mask(self, shape: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """_summary_

        Args:
            shape (Sequence[int]): Shape of the k-space to subsample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]: A 3-tuple containing
            1) the mask for the center of k-space,
            2) the mask for the high frequencies of k-space, and
            3) the integer count of low frequency samples.
        """

        num_cols = shape[-2]  # width
        center_fraction, total_acqusi_step = self.choose_acquisition_step(num_cols)

        num_low_frequencies = round(num_cols * center_fraction)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(num_cols, total_acqusi_step, num_low_frequencies), shape,
        )

        return center_mask, acceleration_mask, num_low_frequencies


def create_full_acquisition_mask(mask_initial_accelerations, center_fractions, final_accelerations, mask_type, seed):

    if mask_type == 'random':
        return AllStepMaskFunc(center_fractions, [mask_initial_accelerations, final_accelerations], seed=seed)
    elif mask_type == 'vds':
        return VariableDensityMask(center_fractions, [mask_initial_accelerations, final_accelerations], seed=seed)


class AllStep_MaskFunc:
    # Add minimal implementation if needed
    pass

class VariableDensityMask:

    def __init__(self, center_fractions_range: Sequence[float], acquisition_range: Sequence[int],
                 allow_any_combination: bool = True, seed: Optional[int] = None):
        """

        Args:
            center_fractions (Sequence[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly each time.
            accelerations (Sequence[int]): Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided, then one of these is chosen uniformly each time.
            allow_any_combination (bool, optional): Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``. Defaults to True.
            seed (Optional[int], optional): Seed for starting the internal random number generator of the
                ``MaskFunc``. Defaults to None.
        """

        # assert len(center_fractions) == len(
        #     accelerations), 'Number of center fractions should match the number of accelerations.'

        self.center_fractions_range = center_fractions_range
        self.acquisition_range = acquisition_range
        self.allow_any_combination = allow_any_combination
        self.rng = np.random.RandomState(seed)
        self.center_fraction, self.acceleration = self.choose_acquisition_step()


    def choose_acquisition_step(self):

        acquisition_fraction_range = np.linspace(1/self.acquisition_range[0], 1/self.acquisition_range[1], 100)
        # random combination
        if self.allow_any_combination:
            return self.rng.choice(self.center_fractions_range), self.rng.choice(acquisition_fraction_range)
        # choose according to the same index
        else:
            choice = self.rng.randint(len(self.center_fractions_range))
            return self.center_fractions_range[choice], self.accelerations[choice]


    def __call__(self, shape: Sequence[int],seed: Optional[Union[int, Tuple[int, ...]]] = None) -> Tuple[torch.Tensor, int]:
        """
        Sample and return a k-space mask

        Args:
            shape (Sequence[int]): Shape of the mask to be created.
            seed (Optional[Union[int, Tuple[int, ...]]], optional): seed for RNG for reproducibility . Defaults to None.

        Returns:
            Tuple[torch.Tensor, int]: A 2-tuple containing
            1) the k-space mask and
            2) the number of center frequency lines
        """

        assert len(shape) >= 3, "Shape should have 3 or more dimensions"

        with temp_seed(self.rng, seed):
            mask = self.get_mask(shape)
        return mask, round(shape[-2] * self.center_fraction)

    def get_prior(self, remaining_indices: Sequence[int]) -> np.ndarray:

            n_cols = len(remaining_indices)

            if n_cols % 2 == 0:
                dist = np.arange(1, n_cols // 2 + 1)
                dist = np.r_[dist, dist[::-1]]
            else:
                dist = np.arange(1, n_cols // 2 + 2)
                dist = np.r_[dist, dist[::-1][:-1]]
            return dist / dist.sum()

    def get_acceleration_mask(
            self, shape, center_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        start = 0
        end = shape[-2] - 1
        len_sampled_indices = end - start + 1

        remaining_indices = set(np.arange(shape[-2]))
        center_mask_indices = set()
        if center_mask is not None:
            center_mask_indices = set(torch.where(center_mask != 0)[0].cpu().numpy())
        remaining_indices -= center_mask_indices

        remaining_indices = {i for i in remaining_indices if i >= start and i <= end}

        num_random_indices = round(len_sampled_indices * self.acceleration) - len(
            center_mask_indices
        )
        num_random_indices = max(0, num_random_indices)

        num_cols = shape[-2]
        mask = torch.zeros(num_cols).float()

        # if there are some mask indices to fill in, fill them in randomly

        if num_random_indices > 0:
            sorted_remaining_indices = sorted(list(remaining_indices))
            sampling_dist = self.get_prior(sorted_remaining_indices)

            random_indices = torch.Tensor(
                np.random.choice(
                    sorted_remaining_indices,
                    num_random_indices,
                    replace=False,
                    p=sampling_dist,
                )
            ).long()

            mask[random_indices] = 1

        return mask

    def reshape_mask(self, mask, shape):
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols  # [1, width, 1]

        return mask.reshape(mask_shape)

    def get_mask(self, shape) -> torch.Tensor:

        center_mask = self.get_center_mask(shape)
        acceleration_mask = self.get_acceleration_mask(shape, center_mask)
        center_mask = center_mask.to(acceleration_mask)
        acceleration_mask = self.reshape_mask(acceleration_mask, shape)
        center_mask = self.reshape_mask(center_mask, shape)
        mask = torch.max(center_mask, acceleration_mask)
        return mask


    def get_center_mask(self, shape) -> torch.Tensor:

        start = 0
        end = shape[-2] - 1

        len_sampled_indices = end - start + 1
        num_low_frequencies = round(len_sampled_indices * self.center_fraction)

        num_cols = shape[-2]

        mask = torch.zeros(num_cols).float()
        if num_low_frequencies == 0:
            return mask

        pad = (num_cols - num_low_frequencies + 1) // 2
        mask[0 + pad: 0 + pad + num_low_frequencies] = 1
        assert mask.sum() == num_low_frequencies

        return mask


