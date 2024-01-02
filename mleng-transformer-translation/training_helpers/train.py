import copy
from functools import partial
import os
from pathlib import Path
import time
from einops import rearrange
import numpy as np
from tqdm import tqdm

from torch.optim import Adam
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from training_helpers.Optim import ScheduledOptim
from training_helpers.dataset import ParallelLanguageDataset
from training_helpers.model import LanguageTransformer, LanguageTransformerEncoder

from alibi_detect.cd import MMDDriftOnline
from alibi_detect.cd.pytorch import preprocess_drift, UAE
from alibi_detect.saving import save_detector

import mlflow
from mlflow.entities import Metric
import optuna


def train_wrapper(kwargs):
    project_path = str(Path(__file__).resolve().parents[1])

    experiment_tags = {
        "mlflow.note.content": "This experiment contains the models for translating English to French sentences"
    }
    if (
        len(
            mlflow.search_experiments(
                filter_string="attribute.name = 'English to French Translation'"
            )
        )
        == 0
    ):
        mlflow.create_experiment("English to French Translation", tags=experiment_tags)
    mlflow.set_experiment("English to French Translation")

    train_dataset = ParallelLanguageDataset(
        project_path + "/data/processed/en/train.pkl",
        project_path + "/data/processed/fr/train.pkl",
        kwargs["num_tokens"],
        kwargs["max_seq_length"],
    )
    # Set batch_size=1 because all the batching is handled in the ParallelLanguageDataset class
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_dataset = ParallelLanguageDataset(
        project_path + "/data/processed/en/val.pkl",
        project_path + "/data/processed/fr/val.pkl",
        kwargs["num_tokens"],
        kwargs["max_seq_length"],
    )
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    model = LanguageTransformer(
        kwargs["vocab_size"],
        kwargs["d_model"],
        kwargs["nhead"],
        kwargs["num_encoder_layers"],
        kwargs["num_decoder_layers"],
        kwargs["dim_feedforward"],
        kwargs["max_seq_length"],
        kwargs["pos_dropout"],
        kwargs["trans_dropout"],
    ).to(kwargs["device"])

    # Use Xavier normal initialization in the transformer
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)

    optim = ScheduledOptim(
        Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        kwargs["init_lr"],
        kwargs["n_warmup_steps"],
    )

    # Use cross entropy loss, ignoring any padding
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    with mlflow.start_run(tags={"group_id": kwargs["group_id"]}) as run:
        mlflow.log_params(kwargs)
        mlflow.log_artifact(
            project_path + "/data/processed/en/train.pkl", "model/files/en"
        )
        mlflow.log_artifact(
            project_path + "/data/processed/en/val.pkl", "model/files/en"
        )
        mlflow.log_artifact(
            project_path + "/data/processed/en/freq_list.pkl", "model/files/en"
        )
        mlflow.log_artifact(
            project_path + "/data/processed/fr/train.pkl", "model/files/fr"
        )
        mlflow.log_artifact(
            project_path + "/data/processed/fr/val.pkl", "model/files/fr"
        )
        mlflow.log_artifact(
            project_path + "/data/processed/fr/freq_list.pkl", "model/files/fr"
        )

        val_loss_final, best_model = train(
            train_loader,
            valid_loader,
            model,
            optim,
            criterion,
            kwargs["num_epochs"],
            kwargs["device"],
            kwargs["trial"] if "trial" in kwargs else None,
            kwargs["vocab_size"],
            kwargs["max_seq_length"],
            run.info.run_id,
        )

        train_data_drift_model(
            best_model, train_dataset, kwargs["max_seq_length"], kwargs["device"]
        )
    return val_loss_final


def train(
    train_loader,
    valid_loader,
    model,
    optim,
    criterion,
    num_epochs,
    device,
    trial,
    vocab_size,
    max_seq_length,
    run_id,
):
    model.train()

    lowest_val = 1e9
    best_model = model
    train_losses = []
    val_losses = []
    total_step = 0
    for epoch in range(num_epochs):
        if not trial:
            pbar = tqdm(total=len(train_loader), leave=False)
        total_loss = 0
        mlflow.log_metric("epoch", epoch + 1, total_step)
        log_batch_train_losses = []

        # Shuffle batches every epoch
        train_loader.dataset.shuffle_batches()
        for step, (src, src_key_padding_mask, tgt, tgt_key_padding_mask) in enumerate(
            iter(train_loader)
        ):
            total_step += 1

            # Send the batches and key_padding_masks to gpu
            src, src_key_padding_mask = src[0].to(device), src_key_padding_mask[0].to(
                device
            )
            tgt, tgt_key_padding_mask = tgt[0].to(device), tgt_key_padding_mask[0].to(
                device
            )
            memory_key_padding_mask = src_key_padding_mask.clone()

            # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)
            tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
            tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to(device)

            # Forward
            optim.zero_grad()
            outputs = model(
                src,
                tgt_inp,
                src_key_padding_mask,
                tgt_key_padding_mask[:, :-1],
                memory_key_padding_mask,
                tgt_mask,
            )
            loss = criterion(
                rearrange(outputs, "b t v -> (b t) v"),
                rearrange(tgt_out, "b o -> (b o)"),
            )

            # Backpropagate and update optim
            loss.backward()
            optim.step_and_update_lr()

            total_loss += loss.item()
            train_losses.append((step, loss.item()))
            log_batch_train_losses.append(
                Metric(
                    key="train_loss",
                    value=loss.item(),
                    step=total_step,
                    timestamp=round(time.time() * 1000),
                )
            )
            if not trial:
                pbar.update(1)
            if step == len(train_loader) - 1:
                if not trial:
                    pbar.close()
                print(
                    f"Epoch [{epoch + 1} / {num_epochs}] \t Step [{step + 1} / {len(train_loader)}] \t "
                    f"Train Loss: {total_loss / len(train_loader)}"
                )
                total_loss = 0

        # Validate every epoch
        if not trial:
            pbar.close()
        val_loss = validate(valid_loader, model, criterion, device, trial)
        val_losses.append((total_step, val_loss))
        mlflow.tracking.MlflowClient().log_batch(run_id, log_batch_train_losses)
        log_batch_train_losses = []
        mlflow.log_metric("validation_loss", val_loss, total_step)
        if trial is not None:
            trial.report(val_loss, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()
        if val_loss < lowest_val:
            lowest_val = val_loss

            best_model = copy.deepcopy(model)
        print(f"Val Loss: {val_loss}")

    best_model.eval().to("cpu")
    model_quantized = torch.quantization.quantize_dynamic(
        best_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    model_compressed = compress_model(model_quantized, vocab_size, max_seq_length)
    val_loss_final = time_model_evaluation(
        valid_loader, model_compressed, criterion, "cpu", trial, "Compressed model"
    )

    mlflow.log_metric("validation_loss_final", val_loss_final)
    mlflow.pytorch.log_model(best_model, "model_orig")
    mlflow.pytorch.log_model(model_compressed, "model")
    return val_loss_final, best_model


def get_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return int(size)


def time_model_evaluation(valid_loader, model, criterion, device, trial, name):
    st = time.time()
    val_loss = validate(valid_loader, model, criterion, device, trial, False)
    elapsed = time.time() - st
    size = get_size_of_model(model)
    print(f"{name}: Val loss {val_loss:.3f} \t Time {elapsed:.3f} \t Size (MB) {size}")
    return val_loss


def validate(valid_loader, model, criterion, device, trial, train_mode=True):
    if not trial:
        pbar = tqdm(total=len(iter(valid_loader)), leave=False)
    if train_mode:
        model.eval()

    total_loss = 0
    for src, src_key_padding_mask, tgt, tgt_key_padding_mask in iter(valid_loader):
        with torch.no_grad():
            src, src_key_padding_mask = src[0].to(device), src_key_padding_mask[0].to(
                device
            )
            tgt, tgt_key_padding_mask = tgt[0].to(device), tgt_key_padding_mask[0].to(
                device
            )
            memory_key_padding_mask = src_key_padding_mask.clone()
            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:].contiguous()
            tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to(device)

            outputs = model(
                src,
                tgt_inp,
                src_key_padding_mask,
                tgt_key_padding_mask[:, :-1],
                memory_key_padding_mask,
                tgt_mask,
            )
            loss = criterion(
                rearrange(outputs, "b t v -> (b t) v"),
                rearrange(tgt_out, "b o -> (b o)"),
            )

            total_loss += loss.item()
            if not trial:
                pbar.update(1)

    if not trial:
        pbar.close()
    if train_mode:
        model.train()
    return total_loss / len(valid_loader)


def ids_tensor(shape, vocab_size, dtype):
    #  Creates a random int32 tensor of the shape within the vocab size
    return torch.randint(0, vocab_size, size=shape, dtype=dtype, device="cpu")


def compress_model(model, vocab_size, max_seq_length):
    src_ids = ids_tensor((1, max_seq_length), vocab_size, torch.int)
    tgt_ids = ids_tensor((1, max_seq_length), vocab_size, torch.int)
    src_key_padding_mask_ids = ids_tensor((1, max_seq_length), 2, torch.bool)
    tgt_key_padding_mask_ids = ids_tensor((1, max_seq_length), 2, torch.bool)
    memory_key_padding_mask_ids = ids_tensor((1, max_seq_length), 2, torch.bool)
    tgt_mask = ids_tensor((max_seq_length, max_seq_length), 2, torch.bool)
    dummy_input = (
        src_ids,
        tgt_ids,
        src_key_padding_mask_ids,
        tgt_key_padding_mask_ids,
        memory_key_padding_mask_ids,
        tgt_mask,
    )
    traced_model = torch.jit.trace(model, dummy_input)
    return traced_model


def gen_nopeek_mask(length):
    """
    Returns the nopeek mask
            Parameters:
                    length (int): Number of tokens in each sentence in the target batch
            Returns:
                    mask (arr): tgt_mask, looks like [[False, True, True],
                                                     [False, False, True],
                                                     [False, False, False]]
    """
    mask = rearrange(torch.triu(torch.ones(length, length)) == 0, "h w -> w h")

    return mask


def train_data_drift_model(model, train_dataset, max_seq_len, device):
    model_encoder = LanguageTransformerEncoder(model)
    embed_src = model.embed_src
    enc_dim = 32
    shape = (
        max_seq_len,
        embed_src.embedding_dim,
    )

    uae = UAE(input_layer=model_encoder, shape=shape, enc_dim=enc_dim)
    uae.to("cuda")

    X_ref = torch.IntTensor(train_dataset.data_1)
    idx = torch.randperm(X_ref.size(0))
    preprocess_fn = partial(
        preprocess_drift, model=uae, max_len=max_seq_len, batch_size=1000, device="cuda"
    )
    cd = MMDDriftOnline(
        X_ref[idx[:10000]],
        ert=200,
        window_size=50,
        preprocess_fn=preprocess_fn,
        backend="pytorch",
        input_shape=(max_seq_len,),
    )
    save_detector(cd, "detector")
    # mlflow.log_artifact("detector", "data_drift_model")
