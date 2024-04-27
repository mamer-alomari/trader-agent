"""
Script for training Stock Trading Bot.

Usage:
  train.py <train-stock> <val-stock> [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug]

Options:
  --strategy=<strategy>             Q-learning strategy to use for training the network. Options:
                                      `dqn` i.e. Vanilla DQN,
                                      `t-dqn` i.e. DQN with fixed target distribution,
                                      `double-dqn` i.e. DQN with separate network for value estimation. [default: t-dqn]
  --window-size=<window-size>       Size of the n-day window stock data representation
                                    used as the feature vector. [default: 10]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch
                                    during training. [default: 32]
  --episode-count=<episode-count>   Number of trading episodes to use for training. [default: 50]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --pretrained                      Specifies whether to continue training a previously
                                    trained model (reads `model-name`).
  --debug                           Specifies whether to use verbose logs during eval operation.
"""

import logging
import coloredlogs

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_train_result,
    switch_k_backend_device
)
import tqdm
import coloredlogs


def main(train_stock, val_stock, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug", pretrained=False,
         debug=False):
    """ Trains the stock trading bot using Deep Q-Learning.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python train.py --help]
    """
    agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)
    
    train_data = get_stock_data(train_stock)
    val_data = get_stock_data(val_stock)

    initial_offset = val_data[1] - val_data[0]

    

    for episode in range(1, ep_count + 1):
        # parallelize tqdm with `leave=False` to avoid overlapping with logging
        # train_model = tqdm.tqdm if debug else tqdm.tqdm
        # tqdm.tqdm.write(f"Episode {episode}/{ep_count}")
        # use progress_apply instead of apply for pandas dataframe
        # use progress bar for training
        # turn off verbose logging if not debugging
        # make the model training output only a bar
        # @tqdm.tqdm(train_result, total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count))
        train_result = train_model(agent, episode, train_data, ep_count=ep_count,
                                   batch_size=batch_size, window_size=window_size)

        # train_result = train_model(agent, episode, train_data, ep_count=ep_count,
        #                            batch_size=batch_size, window_size=window_size)
        val_result, _ = evaluate_model(agent, val_data, window_size, debug)
        show_train_result(train_result, val_result, initial_offset)


if __name__ == "__main__":
    # args = docopt(__doc__)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-stock', type=str, default='AAPL')
    parser.add_argument('--val-stock', type=str, default='AAPL')
    parser.add_argument('--strategy', type=str, default='t-dqn')
    parser.add_argument('--window-size', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--episode-count', type=int, default=2)
    parser.add_argument('--model-name', type=str, default='model_3')
    parser.add_argument('--pretrained', type= bool , default=False)
    parser.add_argument('--debug',  type= bool  , default=False)

    args = parser.parse_args()

    train_stock = args.train_stock
    val_stock = args.val_stock
    strategy = args.strategy
    window_size = args.window_size
    batch_size = args.batch_size
    ep_count = args.episode_count
    model_name = args.model_name
    pretrained = args.pretrained
    debug = args.debug

    # train_stock =     #args.train #args["<train-stock>"]
    # val_stock = args["<val-stock>"]
    # strategy = args["--strategy"]
    # window_size = int(args["--window-size"])
    # batch_size = int(args["--batch-size"])
    # ep_count = int(args["--episode-count"])
    # model_name = args["--model-name"]
    # pretrained = args["--pretrained"]
    # debug = args["--debug"]

    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    try:
        main(train_stock, val_stock, window_size, batch_size,
             ep_count, strategy=strategy, model_name=model_name, 
             pretrained=pretrained, debug=debug)
    except KeyboardInterrupt:
        print("Aborted!")
