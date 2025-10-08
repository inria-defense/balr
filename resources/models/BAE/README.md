# Recipe

This recipe describes the training of the binary attribute encoder model `BAE_mse.pt`.

It includes a description of:

1. the datasets
2. the model used to extract the embedding vectors
3. the configuration parameters used for the training of the BAE

## Datasets

The [Voxceleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) dataset is the corpus used for the training of the BAE. It contains audio samples for various speakers and is split into two sets: `dev` and `test`.

| **name**           | **# of utterances** | **# of speakers** |
| ------------------ | ------------------- | ----------------- |
| voxceleb2-dev.csv  | 1092009             | 5994              |
| voxceleb2-test.csv | 36236               | 118               |

We display here the first few lines of the csv files describing the dataset:

```shell
> head -3 voxceleb2-dev.csv
id,audio,speaker
und/BRY/00000/voxceleb2_SPEECH_und_BRY_00000_03145_5DdGz9cnGw8-10_B8-CPCM16_M10,/lustre/fsn1/projects/rech/idi/commun/corpus/voxceleb2/audio/und/BRY/00000/voxceleb2_SPEECH_und_BRY_00000_03145_5DdGz9cnGw8-10_B8-CPCM16_M10.flac,3145
und/BRY/00000/voxceleb2_SPEECH_und_BRY_00000_03145_5DdGz9cnGw8-8_B8-CPCM16_M10,/lustre/fsn1/projects/rech/idi/commun/corpus/voxceleb2/audio/und/BRY/00000/voxceleb2_SPEECH_und_BRY_00000_03145_5DdGz9cnGw8-8_B8-CPCM16_M10.flac,3145
```

Each line of the csv contains a unique id for the sample, a path to the audio file, and a speaker id.

## Embedding extraction with wespeaker-voxceleb-resnet34-LM

Speaker embeddings for both the `dev` and `test` sets were extracted from the audio samples using Wespeaker's `wespeaker-voxceleb-resnet34-LM` model. To reproduce the extraction, first set the environment variables pointing to the CSV metadata file and output directory for the set. For example, for the `voxceleb2-dev` set,

```shell
export VOXCELEB2_DEV_METADATA=./voxceleb2-dev.csv
export VOXCELEB2_DEV_EMBEDDINGS=./data/voxceleb2-dev/embeddings
```

Then, run the following command to extract embeddings using BALR's CLI command `extract`:

```shell
balr extract --save-dir $VOXCELEB2_DEV_EMBEDDINGS --device cuda --save-output $VOXCELEB2_DEV_METADATA embeddings.model.model_repo=Wespeaker/wespeaker-voxceleb-resnet34-LM embeddings.model.model_name=avg_models
```

Reproduce the same commands for the `voxceleb2-test` set.

## Training the Binary Attribute Encoder

The binary attribute encoder is trained on the embedding vectors extracted from the Voxceleb2 dataset. We used a batch size of `512`, composed of 32 speakers with 16 samples per speaker. We trained it for `200` epochs with a `MSE` loss for optimization. The `dev` set is used for training, while the `test` set is used for validation.

> **Note: If the embeddings were not saved in the same directory as the audio files, update the csv metadata files with a column pointing to the embedding file for each sample.**

To run the training on a single gpu, run the following command

```shell
balr train --save-dir train/mse --device cuda $VOXCELEB2_DEV_METADATA $VOXCELEB2_TEST_METADATA  trainer.epochs=200 'trainer.losses=[mse]' data.N_classes_per_batch=32 data.M_samples_per_class=16 data.batch_size=512 data.num_workers=16
```

## Training logs

To view the metrics logged during the traing of the model, run

```shell
tensorboard --logdir ./resources/models/BAE
```
