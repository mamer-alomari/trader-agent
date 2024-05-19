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
#
# class tkinterGui:
#     def __init__(self):
#         self.root = Tk()
#     self.root.title("Stock Trading Bot Trainer")
#     # get input from user
#     # make train stock and val stock browse buttons to select the stock data files
#     self.train_stock_button = Button(self.root, text="Browse", command=self.get_train_stock)
#     self.val_stock_button = Button(self.root, text="Browse", command=self.get_val_stock)
#     self.train_stock_button.pack()
#     self.val_stock_button.pack()
#     self.strategy_options = ['dqn', 't-dqn', 'double-dqn']
#     self.strategy = tkinter.ttk.Combobox(self.root, values=self.strategy_options)
#
#     self.window_size_options = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#     self.window_size = tkinter.ttk.Combobox(self.root, values=self.window_size_options)
#     self.batch_size_options = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#     self.batch_size = tkinter.ttk.Combobox(self.root, values=self.batch_size_options)
#     # self.batch_size = simpledialog.askinteger("Input", "Enter the batch size for training:", parent=self.root)
#     # make episode count a drop down element between 10 and 100 and increment by 10
#     self.episode_count_options = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#     self.episode_count = tkinter.ttk.Combobox(self.root, values=self.episode_count_options)
#
#     self.episode_count = simpledialog.askinteger("Input", "Enter the number of episodes to train on:", parent=self.root)
#     # make model name a text input element
#     self.model_name = tkinter.Entry(self.root)
#     # Pack all elements
#     self.strategy.pack()
#     self.window_size.pack()
#     self.batch_size.pack()
#     self.episode_count.pack()
#     self.model_name.pack()
#     # get user input for pretrained and debug
#     self.debug = tkinter.IntVar()
#     self.pretrained = tkinter.IntVar()
#     self.root.withdraw()

    # Load stock data

import logging
import coloredlogs
import pandas as pd

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

def iterate_over_data(data):
    for i in range(len(data)):
        yield data[i]
def chunk_data(data, chunk_size):

    chunks =  len(data) // chunk_size


    beggining_indexs =[]
    end_idexs = []
    for i in range(chunks):
        beggining_indexs.append(i*chunk_size)
        end_idexs.append(i*chunk_size + chunk_size)

    return beggining_indexs, end_idexs


def get_best_model(model_results : dict):

    """
    Get the best model from the model results dictionary.
    :param model_results:
    :return:
    """

    best_model = min(model_results, key=model_results.get)
    print (f"Best model is {best_model} with a profit of {model_results[best_model]}")
    # best_model = best_model.split
    return best_model

class chunked_train_model:
    def __init__(self, train_stock, val_stock , model_name, ep_count=100, batch_size=32, window_size=10, debug=False , chunk_size=40):
        self.episode = 1
        self.model_name = model_name
        self.data = get_stock_data(train_stock)
        self.val_data = get_stock_data(val_stock)
        self.ep_count = ep_count
        self.batch_size = batch_size
        self.window_size = window_size
        self.debug = debug
        self.chunk_size = chunk_size
        self.model_name= None
        self.chuncking_indexes, self.ending_indexes = chunk_data(self.data, self.chunk_size)
        self.agent = None
        self.best_model=None
        self.model_results = {}
        self.model_history = {}
        self.loss_dict = {}
        self.chunk_number = 1


    def train_model_using_best_previous_model(self ,model_name):
        # self.episode = episode
        # self.ep_count = ep_count
        # self.model_name = model_name
        # self.data = get_stock_data(train_stock)
        # self.model_name = model_name

        for episode in range(1, self.ep_count + 1):
            # self.episode = episode
            print ("###################################################")
            print (f"Currently on episode {episode} of {self.ep_count}")
            print ("###################################################")
            if episode < 3 :
                print(f'Currently on episode {episode} of {self.ep_count}')
                print(f'Currently on chunk {self.data} of {len(self.data)}')
                self.model_name = f"{model_name}_{str(episode)}_{self.chunk_number}"
                self.agent = Agent(self.window_size, strategy=strategy, pretrained=False, model_name=self.model_name)
            else:
                self.best_model = get_best_model(self.loss_dict)
                self.agent = Agent(self.window_size, strategy=strategy, pretrained=True, model_name=self.best_model)

            for i in range(len(self.chuncking_indexes)):

                initial_offset = self.val_data[1] - self.val_data[0]
                self.chunk_number = i
                self.model_name = f"{model_name}_{str(episode)}_{self.chunk_number}"
                self.chunk=self.data[self.chuncking_indexes[i]:self.ending_indexes[i]]

                if self.best_model is not None:
                    self.model_name = self.best_model


                train_result=train_model(self.agent, self.episode, self.chunk, ep_count=ep_count, batch_size=self.batch_size,
                                         window_size=self.window_size,model_name=self.model_name)
                loss = train_result[3]
                self.loss_dict[self.model_name] = float(loss)
                with open(f'logs/loss.txt', 'a') as f:
                    f.write(f'{self.model_name} \n'
                            f'{self.loss_dict[self.model_name]} \n')
                print ("#################################")
                print(f"Loss: {loss}")
                print ("#################################")
                val_result, history = evaluate_model(self.agent, self.val_data, self.window_size , self.debug)
                print("############################################")
                print(f"Val result: {val_result}")
                print("############################################")
                self.model_results[self.model_name] = val_result
                self.model_history[self.model_name] = history
                print (f"Model results: {self.model_results.keys()}")

                with open(f'logs/model_results.txt', 'a') as f:
                    f.write(f'{self.model_name} \n' 
                            f'{self.model_results[self.model_name]} \n')
                with open(f'logs/model_history.txt', 'a') as f:
                    f.write(f'{self.model_name} \n'
                            f'{self.model_history[self.model_name]} \n')



        return self.model_results, self.model_history , self.loss_dict


def main(train_stock, val_stock, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug",chunk_size=450 , pretrained=False,
         debug=False , incrementally_train=True):
    """ Trains the stock trading bot using Deep Q-Learning.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python train.py --help]
    """

    trainer = chunked_train_model(train_stock, val_stock,model_name, ep_count=ep_count,
                                  batch_size=batch_size, window_size=window_size, debug=debug, chunk_size=chunk_size)

    trainer.train_model_using_best_previous_model(model_name)

    # for episode in range(1, ep_count + 1):
    #
    #
    #     ###################################### old code #########################################
    #     if incrementally_train == False:
    #         agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)
    #         train_result = train_model(agent, episode, train_data, ep_count=ep_count,
    #                                    batch_size=batch_size, window_size=window_size)
    #         val_result, _ = evaluate_model(agent, val_data, window_size, debug)
    #         # show_train_result(train_result, val_result, initial_offset , model_name=model_name)
    #
    #         with open('train.log', 'a') as f:
    #             f.write((f'Episode {str(train_result[0])}/{str(train_result[1])} - '
    #                      f'Train Position: {str(train_result(val_result[2]))}  Val Position: {str(format_position(val_result))}  '
    #                      f'Train Loss: {str(val_result[3])}' + '\n'
    #                      .format(train_result[0], train_result[1], train_result(val_result[2]), format_position(val_result), val_result[3],)))
    #


if __name__ == "__main__":
    # args = docopt(__doc__)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-stock', type=str, default='AAPL')
    parser.add_argument('--val-stock', type=str, default='AAPL')
    parser.add_argument('--strategy', type=str, default='t-dqn')
    parser.add_argument('--window-size', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=480)
    parser.add_argument('--chunk-size', type=int, default=500)
    parser.add_argument('--episode-count', type=int, default=8)
    parser.add_argument('--model-name', type=str, default='model_3')
    parser.add_argument('--pretrained', type= bool , default=False)
    parser.add_argument('--debug',  type= bool  , default=False)




    args = parser.parse_args()

    train_stock = args.train_stock
    val_stock = args.val_stock
    strategy = args.strategy
    window_size = args.window_size
    batch_size = args.batch_size
    chunk_size = args.chunk_size
    ep_count = args.episode_count
    model_name = args.model_name
    pretrained = args.pretrained
    debug = args.debug


    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    try:
        main(train_stock, val_stock, window_size, batch_size,
             ep_count, strategy=strategy, model_name=model_name, chunk_size=chunk_size,
             pretrained=pretrained)
    except KeyboardInterrupt:
        print("Aborted!")


