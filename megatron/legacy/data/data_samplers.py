# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Dataloaders."""


import random
import torch
import numpy as np
from torch.utils.data import Dataset
from megatron.training import get_args
from megatron.core import mpu


def build_pretraining_data_loader(dataset, consumed_samples):
    """Build dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()

    # Megatron sampler
    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding)
    elif args.dataloader_type == "external":
        # External dataloaders are passed through. User is expected to provide a
        # torch-compatible dataloader and define samplers, if needed.
        return dataset
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       persistent_workers=True if args.num_workers > 0 else False,
                                       )

class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class RandomSeedDataset(Dataset):

    def __init__(self, dataset):
        args = get_args()
        self.base_seed = args.seed
        self.curr_seed = args.seed
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.curr_seed = self.base_seed + epoch

    def __getitem__(self, idx):
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]


class MegatronPretrainingRandomSampler:

    def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, data_sharding):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        args = get_args()
        orin_data_seed = getattr(args, "orin_data_seed", None)
        self.seed = int(args.seed if orin_data_seed is None else orin_data_seed)
        self.seed_mode = getattr(args, "orin_seed_mode", "megatron")
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def _llamafactory_rank_batches(self, idx_range_total, local_batches_to_skip):
        """Yield batches using HF Trainer/Accelerate BatchSamplerShard semantics."""

        initial_data = []
        batch_to_yield = []
        emitted = 0
        idx = -1
        batch = []
        for idx, start in enumerate(range(0, self.total_samples, self.micro_batch_size)):
            batch = idx_range_total[start : start + self.micro_batch_size]
            if idx < self.data_parallel_size:
                initial_data += batch
            if idx % self.data_parallel_size == self.data_parallel_rank:
                batch_to_yield = batch
            if idx % self.data_parallel_size == self.data_parallel_size - 1 and len(batch) == self.micro_batch_size:
                if emitted >= local_batches_to_skip:
                    yield batch_to_yield
                emitted += 1
                batch_to_yield = []

        if not initial_data:
            return

        if len(batch_to_yield) == self.micro_batch_size:
            if emitted >= local_batches_to_skip:
                yield batch_to_yield
            emitted += 1

        while len(initial_data) < self.micro_batch_times_data_parallel_size:
            initial_data += initial_data

        if len(batch) == self.micro_batch_size:
            batch = []
            idx += 1

        cycle_index = 0
        while idx % self.data_parallel_size != 0 or len(batch) > 0:
            end_index = cycle_index + self.micro_batch_size - len(batch)
            batch += initial_data[cycle_index:end_index]
            if idx % self.data_parallel_size == self.data_parallel_rank:
                if emitted >= local_batches_to_skip:
                    yield batch
                emitted += 1
            cycle_index = end_index
            batch = []
            idx += 1

    def __iter__(self):
        if self.seed_mode == "llamafactory":
            if self.consumed_samples > 0:
                assert self.consumed_samples % self.micro_batch_times_data_parallel_size == 0
            total_batches = (self.total_samples + self.micro_batch_size - 1) // self.micro_batch_size
            rank_batches_per_epoch = (
                total_batches + self.data_parallel_size - 1
            ) // self.data_parallel_size
            epoch_samples = rank_batches_per_epoch * self.micro_batch_times_data_parallel_size
            self.epoch = self.consumed_samples // epoch_samples
            current_epoch_samples = self.consumed_samples % epoch_samples
            local_batches_to_skip = current_epoch_samples // self.micro_batch_times_data_parallel_size

            if isinstance(self.dataset, RandomSeedDataset):
                self.dataset.set_epoch(self.epoch)
            shuffle_seed = self.seed + self.epoch

            g = torch.Generator()
            g.manual_seed(shuffle_seed)
            idx_range_total = torch.randperm(self.total_samples, generator=g).tolist()
            for batch in self._llamafactory_rank_batches(idx_range_total, local_batches_to_skip):
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
            return

        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)
        shuffle_seed = self.seed + self.epoch

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                           * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(shuffle_seed)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) \
                                * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(shuffle_seed)
            idx_range_total = \
                torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank::self.data_parallel_size]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []
