import click
import torch
from trl import DPOTrainer
from configs import get_configs
from gpt import GPTActor, GPTRewardModel, GPTCritic
from dataset import DahoasSFTStaticPromptsDataset


def train(batch_size, exp_name, model_weights):
    cfg = get_configs("gpt2-medium")
    cfg.model_weights = model_weights
    cfg.sft_model_weights = cfg.model_weights
    cfg.batch_size = batch_size
    cfg.total_epochs = 2
    cfg.exp_name = exp_name

    model = GPTActor.from_checkpoint(cfg, cfg.model_weights).cuda()
    sft_model_ref = GPTActor.from_checkpoint(cfg, cfg.sft_model_weights).cuda()


    dataset = None # TODO Replace with our dataset
    # TODO Make a new wrapper class (in trainers.py) which mimics other's interface but wraps this DPOTrainer
    tokenizer = TiktokenTokenizer('gpt2')
    trainer = DPOTrainer(
        model,
        sft_model_ref,
        # args
        beta = 0.1,
        train_dataset=dataset,
        tokenizer = tokenizer
    )
    trainer.train()
    trainer.save_model('./output_dir')


@click.command()
@click.option('--strategy', '-s')
@click.option('--batch-size', '-b', default=1)
@click.option('--exp-name', '-n', default="default")
@click.option('--model', '-m')
def main(strategy, batch_size, exp_name, model):
    train(batch_size, exp_name, model)


if __name__ == "__main__":
    main()