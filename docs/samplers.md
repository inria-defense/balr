# Samplers

The BA-LR toolkit provides [`torch.utils.data.Samplers`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) to enforce certain conditions on the batches used for [training a Binary Attribute Encoder](./trainer.md).

## NxMSampler

A `NxMSampler` samples a dataset in batches where each batch of size N * M contains N classes and M samples per class. Different implementations of the NxMSampler can enforce various extra conditions, such as ensuring that each batch contains N distinct classes.

!!! warning

    A dataset used with a `NxMSampler` must have at least N distinct classes in it.

## RandomNxMSampler

An implementation of `NxMSampler` that only samples each individual class once, always sampling M samples per class. Thus, the length of the sampler is equal to **the number of distinct classes multiplied by M**. If the number of distinct classes is not divisible by N, either the extra classes are omitted if `drop_last` is True, or additional classes are sampled again otherwise.

Each batch in a `RandomNxMSampler` will have **samples from N classes that are distinct**. The M samples for an individual class are sampled randomly from all the samples for that class present in the dataset. If there are less samples for a class than M, the samples for that class will be repeated.

For example, for a dataset with 10 samples and 3 classes

```python
labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
sampler = RandomNxMSampler(labels, N_classes_per_batch=2, M_samples_per_class=2, shuffle=False, drop_last=True)
list(sampler) # [3, 4, 5, 7]
```

the `RandomNxMSampler` with `N=2` and `M=2` and `drop_last=True` will return 1 batch of 2*2 samples (samples for each class are picked at ramdom) from the first two classes (the last class will be droped).

If `drop_last=False`, the sampler will return 2 batches of 2*2 samples, the second batch having 2 samples from the last class + 2 samples from the first class again.

```python
labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
sampler = RandomNxMSampler(labels, N_classes_per_batch=2, M_samples_per_class=2, shuffle=False, drop_last=False)
list(sampler) # [2, 0, 6, 5, 9, 8, 0, 3]
```

!!! note

    If `shuffle=True`, the order the classes are sampled in is random.

## ExhaustiveNxMSampler

An implementation of `NxMSampler` that will iterate over elements of the dataset by batches of N * M samples, until either all the dataset has been sampled, or there are less than N * M samples remaining.

As much as possible, the batches will have N distinct classes, **but this is not guaranteed, for instance if there are only samples from a single class remaining**.

Samples for a class might be repeated if the number of samples for that class is not divisible by `M`.

For example, for a dataset with 10 samples and 3 classes

```python
labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
sampler = ExhaustiveNxMSampler(labels, N_classes_per_batch=2, M_samples_per_class=2, shuffle=False)
list(sampler) # [0, 1, 5, 6, 8, 9, 2, 3, 7, 5, 4, 0]
```

the `ExhaustiveNxMSampler` with `N=2` and `M=2` will return 3 batch of 2*2 samples. Classes `0` and `1` have 5 and 3 samples respectively, which is not divisible by `M=2`, so both these classes will have their first item sampled twice.

* The first batch (`0, 1, 5, 6`) contains the first 2 samples from class `0` (indices `0` and `1`) and the first two samples from class `1` (indices `5` and `6`)
* The second batch (`8, 9, 2, 3`) contains the first 2 samples from class `2` (indices `8` and `9`) and two more samples from class `0` (indices `2` and `3`)
* The last batch (`7, 5, 4, 0`) contains the last sample from class `1` (index `7`) plus an extra sample from the same class, which has already been sampled but is required to have `M=2` samples per class (index `5`), and also the last sample from class `0` (index `4`) plus another sample from the same class that has already been sampled (index `0`)

!!! note

    If `shuffle=True`, the order the classes are sampled in is random, and the order of the samples for each class is also random.


## Distributed sampling

The `NxMSampler` implementations support distributed sampling for training with multiple gpus. The samplers will assign the batches to each replica in turn (the first batch of size N * M goes to the first gpu, the second batch to the second gpu, the third batch to the first, and so on...). To make sure that each replica receives the same amount of data, the first few batches might be repeated (for exemple, if there are 3 batches of size N * M and 2 gpus, the first gpu will receive the first and third batches, while the second gpu will receive the second and first batches).

!!! warning

    In distributed mode, as with pytorch's [`DistributedSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler), calling the `set_epoch` method on the sampler at the beginning of each epoch **before** creating the `DataLoader` iterator is necessary to make shuffling work properly, otherwise the same ordering will be always used.

## CLI

The sampler class used during [training of the Binary Attribute Encoder](./trainer.md) is set in the `data_config` config parameter. By default, the `RandomNxMSampler` is used. To use another sampler, set the `sampler` attribute of the data configuration to the class path of the sampler to use:

!!! example

    ```bash
    balr train resources/data/voxceleb2/train.csv resources/data/voxceleb2/test.csv data.sampler=balr.samplers.nxm_samplers.ExhaustiveNxMSampler
    ```
