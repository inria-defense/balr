from collections.abc import Generator
from unittest.mock import ANY, DEFAULT, MagicMock, PropertyMock, call, patch

import pytest
import torch
from sklearn.calibration import LabelEncoder
from torch import nn, optim

from balr.binary_attributes.autoencoder import AutoEncoder
from balr.binary_attributes.trainer import BinaryAttributeEncoderTrainer
from balr.config.configs import DataConfig
from balr.data.dataset import AudioDataset
from balr.losses.arcface_loss import ArcFaceLoss
from balr.losses.mse_loss import MSELoss
from balr.losses.triplet_margin_loss import TripletMarginLoss
from tests.conftest import ROOT_DIR


@pytest.fixture(scope="session", autouse=True)
def patch_trainer_writer() -> Generator[None]:
    """
    Patch BinaryAttributeEncoderTrainer's _setup_writers method to avoid creating
    runs/train folder during tests.
    """
    with patch(
        "balr.binary_attributes.trainer.BinaryAttributeEncoderTrainer._setup_writers"
    ):
        yield


@pytest.fixture
def trainset() -> AudioDataset:
    dataset = AudioDataset.from_csv(ROOT_DIR / "resources/data/voxceleb2/train.csv")
    return dataset


@pytest.fixture
def testset() -> AudioDataset:
    dataset = AudioDataset.from_csv(ROOT_DIR / "resources/data/voxceleb2/test.csv")
    return dataset


@patch("balr.binary_attributes.trainer.nn.init.kaiming_normal_")
def test_ba_trainer_setup_model(
    mock_kaiming, trainset: AudioDataset, testset: AudioDataset
):
    # mock kaiming init will set initial weigts to constant value
    mock_constant_weight = 333

    def mock_kaiming_init(tensor, mode):
        with torch.no_grad():
            tensor.fill_(mock_constant_weight)

    mock_kaiming.side_effect = mock_kaiming_init

    input_dim = 8
    internal_dim = 16
    trainer = BinaryAttributeEncoderTrainer(
        trainset,
        testset,
        DataConfig(),
        [MSELoss()],
        input_dim=input_dim,
        internal_dim=internal_dim,
    )
    assert trainer.input_dim == input_dim
    assert trainer.internal_dim == internal_dim

    trainer._setup_model(input_dim, internal_dim)

    assert trainer.model is not None
    assert isinstance(trainer.model, AutoEncoder)
    assert trainer.model.input_dim == input_dim
    assert trainer.model.internal_dim == internal_dim
    for module in trainer.model.modules():
        if isinstance(module, nn.Linear):
            assert torch.all(module.weight == mock_constant_weight)


def test_ba_trainer_setup_loss(trainset: AudioDataset, testset: AudioDataset):
    mse_loss = MagicMock(wraps=MSELoss())
    type(mse_loss).name = PropertyMock(return_value="MSE")
    triplet_loss = MagicMock(wraps=TripletMarginLoss())
    type(triplet_loss).name = PropertyMock(return_value="Triplet")
    trainer = BinaryAttributeEncoderTrainer(
        trainset, testset, DataConfig(M_samples_per_class=19), [mse_loss, triplet_loss]
    )
    trainer.nb_train_classes = 8

    trainer._setup_loss()
    mse_loss.setup.assert_called_once_with(
        input_dim=trainer.input_dim,
        internal_dim=trainer.internal_dim,
        nb_train_classes=trainer.nb_train_classes,
        M_samples_per_class=trainer.M_samples_per_class,
    )
    triplet_loss.setup.assert_called_once_with(
        input_dim=trainer.input_dim,
        internal_dim=trainer.internal_dim,
        nb_train_classes=trainer.nb_train_classes,
        M_samples_per_class=trainer.M_samples_per_class,
    )

    assert trainer.loss_labels == ("MSE", "Triplet")


def test_ba_trainer_build_optimizer(trainset: AudioDataset, testset: AudioDataset):
    learning_rate = 0.009876
    trainer = BinaryAttributeEncoderTrainer(
        trainset, testset, DataConfig(), [MSELoss()], learning_rate=learning_rate
    )

    with pytest.raises(RuntimeError, match="Model not initialized."):
        trainer._build_optimizer()

    trainer._setup_model(256, 512)
    trainer._build_optimizer()
    assert isinstance(trainer.optimizer, optim.Adam)
    assert len(trainer.optimizer.param_groups[0]["params"]) == len(
        list(trainer.model.parameters())
    )

    arcface_loss = ArcFaceLoss()
    trainer = BinaryAttributeEncoderTrainer(
        trainset,
        testset,
        DataConfig(),
        [MSELoss(), arcface_loss],
        learning_rate=learning_rate,
    )
    trainer._setup_model(256, 512)
    trainer._build_optimizer()
    assert len(trainer.optimizer.param_groups[0]["params"]) == len(
        list(trainer.model.parameters())
    ) + len(list(arcface_loss.parameters()))


def test_ba_trainer_get_dataloader(
    voxceleb2_embedding_files, trainset: AudioDataset, testset: AudioDataset
):
    data_config = DataConfig(sampler=None, batch_size=4, shuffle=False)
    trainer = BinaryAttributeEncoderTrainer(trainset, testset, data_config, [MSELoss()])

    train_loader = trainer.get_dataloader(trainset, data_config, -1, "train")

    speakers = [item["speaker"] for _, item in trainset.iter_dicts()]
    encoder = LabelEncoder()
    speaker_labels = encoder.fit_transform(speakers)
    assert trainer.nb_train_classes == len(encoder.classes_)

    # expect train_loader to provide encoded labels in batches
    for batch_id, (_, labels) in enumerate(train_loader):
        for i in range(len(labels)):
            idx = batch_id * data_config.batch_size + i
            expected_label = speaker_labels[idx]
            assert labels[i].item() == expected_label


@patch("balr.binary_attributes.trainer.build_sampler")
def test_ba_trainer_loads_sampler(
    mock_build_sampler, trainset: AudioDataset, testset: AudioDataset
):
    mock_sampler = MagicMock()
    mock_build_sampler.return_value = mock_sampler
    data_config = DataConfig(sampler="mock.sampler.path", batch_size=4, shuffle=False)
    rank = -1
    trainer = BinaryAttributeEncoderTrainer(trainset, testset, data_config, [MSELoss()])

    train_loader = trainer.get_dataloader(trainset, data_config, rank, "train")

    mock_build_sampler.assert_called_once_with(
        data_config.sampler,
        ANY,
        data_config.N_classes_per_batch,
        data_config.M_samples_per_class,
        data_config.batch_size,
        data_config.shuffle,
        False,
        rank,
    )
    assert train_loader.sampler is mock_sampler


def test_ba_trainer_train(trainset: AudioDataset, testset: AudioDataset):
    epochs = 10
    trainer = BinaryAttributeEncoderTrainer(
        trainset,
        testset,
        DataConfig(sampler=None),
        [MSELoss()],
        epochs=epochs,
        val_period=0,
        log_period=0,
    )
    mock_loss = torch.zeros(3)
    trainer.train_loss = mock_loss

    with patch.multiple(
        trainer,
        _run_epoch=DEFAULT,
        validate=DEFAULT,
        _save_metrics=DEFAULT,
        save_model=DEFAULT,
    ) as mock_methods:
        trainer.train()

        # assert _run_epoch is called on every epoch
        assert mock_methods["_run_epoch"].call_count == epochs
        mock_methods["_run_epoch"].assert_has_calls(
            [call(epoch) for epoch in range(epochs)]
        )
        # assert validate is called only on final epoch
        mock_methods["validate"].assert_called_once()
        # assert _save_metrics is not called
        mock_methods["_save_metrics"].assert_not_called()
        # assert save_model is called on every epoch
        assert mock_methods["save_model"].call_count == epochs

    trainer = BinaryAttributeEncoderTrainer(
        trainset,
        testset,
        DataConfig(sampler=None),
        [MSELoss()],
        epochs=epochs,
        val_period=3,
        log_period=4,
    )
    trainer.train_loss = mock_loss

    with patch.multiple(
        trainer,
        _run_epoch=DEFAULT,
        validate=DEFAULT,
        _save_metrics=DEFAULT,
        save_model=DEFAULT,
    ) as mock_methods:
        trainer.train()

        # assert validate is called 4 times
        assert mock_methods["validate"].call_count == 4
        # assert _save_metrics is called 2 times
        assert mock_methods["_save_metrics"].call_count == 2
        mock_methods["_save_metrics"].assert_has_calls(
            [call(mock_loss, 3, "train"), call(mock_loss, 7, "train")]
        )


def test_ba_trainer_run_epoch(
    voxceleb2_embedding_files, trainset: AudioDataset, testset: AudioDataset
):
    batch_size = 2
    mse_weight = torch.tensor([1.2])
    triplet_weight = torch.tensor([0.5])
    trainer = BinaryAttributeEncoderTrainer(
        trainset,
        testset,
        DataConfig(sampler=None, batch_size=batch_size, num_workers=1),
        [MSELoss(weight=mse_weight), TripletMarginLoss(weight=triplet_weight)],
    )
    trainer._setup_train()

    with patch.object(trainer, "_process_batch") as mock_process_batch:
        mock_mse, mock_asl = torch.randn(
            (), dtype=torch.float, requires_grad=True
        ), torch.randn(())
        mock_process_batch.return_value = [mock_mse, mock_asl]
        trainer._run_epoch(4)

        assert mock_process_batch.call_count == len(trainset) / batch_size
        loss = mse_weight * mock_mse + triplet_weight * mock_asl
        assert torch.allclose(
            trainer.train_loss, torch.hstack([loss, mock_mse, mock_asl])
        )


def test_ba_trainer_process_batch(trainset: AudioDataset, testset: AudioDataset):
    features = torch.randn((100, 512))
    labels = torch.randint(0, 7, (100,))
    recon = torch.randn((100, 512))
    binary = torch.randint(0, 1, (512, 2))
    Z = torch.randn((100, 512))
    mse = torch.randn(())
    triplet = torch.randn(())
    mse_loss = MagicMock(wraps=MSELoss())
    mse_loss.return_value = mse
    triplet_loss = MagicMock(wraps=TripletMarginLoss())
    triplet_loss.return_value = triplet
    model_mock = MagicMock()
    model_mock.return_value = recon, binary, Z

    # create trainer with single loss function (mse)
    trainer = BinaryAttributeEncoderTrainer(trainset, testset, DataConfig(), [mse_loss])
    trainer.model = model_mock

    # Call _process_batch and expect only one loss value
    res = trainer._process_batch(features, labels)
    model_mock.assert_called_once_with(features)
    mse_loss.assert_called_once_with(features, labels, binary, recon, Z)
    assert len(res) == 1
    assert res[0] is mse

    # create trainer with two loss functions (mse, triplet)
    trainer = BinaryAttributeEncoderTrainer(
        trainset, testset, DataConfig(), [mse_loss, triplet_loss]
    )
    trainer.model = model_mock
    model_mock.reset_mock()
    mse_loss.reset_mock()

    # Call _process_batch and expect two loss values
    res = trainer._process_batch(features, labels)
    model_mock.assert_called_once_with(features)
    mse_loss.assert_called_once_with(features, labels, binary, recon, Z)
    triplet_loss.assert_called_once_with(features, labels, binary, recon, Z)
    assert len(res) == 2
    assert res[0] is mse
    assert res[1] is triplet
