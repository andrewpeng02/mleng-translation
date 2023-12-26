import click
import mlflow
import optuna

from training_helpers.train import train_wrapper


@click.command()
@click.option("--auto_optimize", type=bool, default=False)
@click.option("--group_id", type=str, default="")
@click.option("--init_lr", type=float, default=0.01)
@click.option("--num_epochs", type=int, default=15)
@click.option("--max_seq_length", type=int, default=96)
@click.option("--num_tokens", type=int, default=8000)
@click.option("--vocab_size", type=int, default=15000 + 4)
@click.option("--d_model", type=int, default=512)
@click.option("--num_encoder_layers", type=int, default=6)
@click.option("--num_decoder_layers", type=int, default=6)
@click.option("--dim_feedforward", type=int, default=2048)
@click.option("--nhead", type=int, default=8)
@click.option("--pos_dropout", type=float, default=0.1)
@click.option("--trans_dropout", type=float, default=0.1)
@click.option("--n_warmup_steps", type=int, default=1500)
@click.option("--device", type=str, default="cuda")
def main(**kwargs):
    if kwargs["auto_optimize"]:
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.SuccessiveHalvingPruner(),
        )

        # try default params first
        study.enqueue_trial(
            {
                "nhead": 3,
                "d_model": 4,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
                "dim_feedforward": 10,
                "pos_dropout": 0.1,
                "trans_dropout": 0.1,
                "n_warmup_steps": 5,
                "init_lr": 0.01,
            }
        )

        def train_wrapper_optimize(trial: optuna.trial.Trial):
            kwargs["nhead"] = 2 ** trial.suggest_int("nhead", 1, 4)  # between 2 and 16
            kwargs["d_model"] = 2 ** (
                5 + trial.suggest_int("d_model", 2, 5)
            )  # between 128 and 1024

            kwargs["num_encoder_layers"] = trial.suggest_int("num_encoder_layers", 2, 8)
            kwargs["num_decoder_layers"] = trial.suggest_int("num_decoder_layers", 2, 8)
            kwargs["dim_feedforward"] = 2 ** trial.suggest_int(
                "dim_feedforward", 7, 12
            )  # between 128 and 4096
            kwargs["pos_dropout"] = trial.suggest_float("pos_dropout", 0.1, 0.4)
            kwargs["trans_dropout"] = trial.suggest_float("trans_dropout", 0.1, 0.4)
            kwargs["n_warmup_steps"] = 300 * trial.suggest_int("n_warmup_steps", 3, 8)
            kwargs["init_lr"] = trial.suggest_float("init_lr", 0.001, 0.1, log=True)
            kwargs["trial"] = trial

            return train_wrapper(kwargs)

        study.optimize(train_wrapper_optimize, n_trials=1)
    else:
        train_wrapper(kwargs)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://0.0.0.0:8000")
    main()
