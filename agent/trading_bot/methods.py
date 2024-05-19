import os
import logging

import numpy as np

from tqdm import tqdm

from .utils import (
    format_currency,
    format_position
)
from .ops import (
    get_state
)


def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=10, model_name='model_name'):

    # agent.inventory = []
    avg_loss = []
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.inventory = []
    last_action = 0
    last_position_price = 0
    length_of_position= 0



    state = get_state(data, 0, window_size + 1)

    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):

            length_of_position += 1
        # for t in range(data_length):
            reward = 0
            next_state = get_state(data, t + 1, window_size + 1)

            # select an action
            action = agent.act(state, is_eval=True)
            if length_of_position // 20 == 1 :
                if action == 1:
                    action = 2
                elif action == 2:
                    action = 1
                else:
                    action = 0
            length_of_position = 0
            # BUY
            if action == 1:
                if len(agent.inventory) > 0 and last_action == 1:
                    # last_action = 1
                    pass
                elif len(agent.inventory) > 0 and last_action == 2:

                    sold_price = agent.inventory.pop(0)
                    delta = data[t] - last_position_price  # sold_price
                    reward = delta
                    total_profit += delta
                    last_position_price = data[t]
                    history.append((data[t], "CLOSED_SELL"))
                    history.append((data[t], "HOLD"))
                    last_action = 0

                else:
                    agent.inventory.append(data[t])
                    history.append((data[t], "BUY"))
                    last_action = 1
                    last_position_price = data[t]

                # agent.inventory.append(data[t])

            # SELL
            elif action == 2 and len(agent.inventory) > 0:
                # same action from last state
                if last_action == 2:  # if the last action was hold or sell pass
                    # last_action = 2
                    pass
                # different action from last state
                elif last_action == 1:  # if the last action was buy
                    bought_price = agent.inventory.pop(0)
                    delta = last_position_price - data[t]  # bought_price - data[t]
                    reward = delta
                    total_profit += delta
                    history.append((data[t], "CLOSED_BUY"))
                    history.append((data[t], "HOLD"))
                    last_action = 0
                    last_position_price = data[t]
                # last state being hold
                else:
                    agent.inventory.append(2)
                    last_position_price = data[t]
                    history.append((last_position_price, "SELL"))
                    last_action = 2

            # HOLD
            else:
                if len(agent.inventory) > 0 and last_action == 0:
                    # last_action = 0
                    reward = -0.5

                elif len(agent.inventory) > 0 and last_action == 1:
                    # agent.inventory.pop(0)
                    bought_price = agent.inventory.pop(0)
                    delta = data[t] - last_position_price  # bought_price
                    reward = delta
                    total_profit += delta
                    history.append((data[t], "CLOSED_BUY"))
                    history.append((data[t], "HOLD"))
                    last_action = 0
                    last_position_price = 0

                elif len(agent.inventory) > 0 and last_action == 2:
                    sold_price = agent.inventory.pop(0)
                    delta = last_position_price - data[t]  # sold_price - data[t]
                    reward = delta
                    total_profit += delta
                    history.append((data[t], "CLOSED_SELL"))
                    history.append((data[t], "HOLD"))
                    last_action = 0
                    last_position_price = 0

                else:
                    agent.inventory.append(0)
                    history.append((data[t], "HOLD"))
                    last_action = 0
                    last_position_price = 0

            if t == data_length - 1:
                if last_action == 1:
                    bought_price = agent.inventory.pop(0)
                    delta = data[t] - last_position_price
                    total_profit += delta
                    history.append((data[t], "CLOSED_BUY"))
                elif last_action == 2:
                    sold_price = agent.inventory.pop(0)
                    delta = last_position_price - data[t]
                    total_profit += delta
                    history.append((data[t], "CLOSED_SELL"))
                else:
                    last_action = 0

                last_position_price = 0

            done = (t == data_length - 1)
            agent.memory.append((state, action, reward, next_state, done))

            if len(agent.memory) > batch_size:
                loss = agent.train_experience_replay(batch_size)
                avg_loss.append(loss)

        # state = next_state

    # if episode % 10 == 0:
    agent.save(model_name)
    with open('logs/train.log', 'a') as f:
        f.write (f"{model_name} \n "
                 f"Episode {episode}/{ep_count} \n"
                 f"Total Profit: {format_position(total_profit)} \n"
                 f"model_history: {history} \n")

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, window_size, debug):
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.inventory = []
    last_action = 0
    last_position_price = 0

    length_of_position = 0
    
    state = get_state(data, 0, window_size + 1)

    for t in range(data_length):
        length_of_position += 1

        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)
        
        # select an action
        action = agent.act(state, is_eval=True)
        if length_of_position // 20 == 1 :
            if action == 1:
                action = 2
            elif action == 2:
                action = 1
            else:
                action = 0
            length_of_position = 0
            # action = 1 if action == 2 , 2 if last_action ==1  else 0

        # BUY
        if action == 1:
            if len(agent.inventory) > 0 and last_action == 1 :
                # last_action = 1
                pass
            elif len(agent.inventory) > 0 and last_action == 2 :

                sold_price = agent.inventory.pop(0)
                delta = data[t] - last_position_price #sold_price
                reward = delta
                total_profit += delta
                last_action = 1
                last_position_price = data[t]
                history.append((data[t], "CLOSED_SELL"))
            else :
                agent.inventory.append(data[t])
                history.append((data[t], "BUY"))
                last_action = 1
                last_position_price = data[t]

            # agent.inventory.append(data[t])

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            # same action from last state
            if last_action == 2 :   # if the last action was hold or sell pass
                # last_action = 2
                pass
            # different action from last state
            elif last_action == 1 :  # if the last action was buy
                bought_price = agent.inventory.pop(0)
                delta = last_position_price - data[t] #bought_price - data[t]
                reward = delta
                total_profit += delta
                history.append((data[t], "CLOSED_BUY"))
                last_action = 2
                last_position_price = data[t]
            # last state being hold
            else :
                agent.inventory.append(2)
                last_position_price = data[t]
                history.append((last_position_price, "SELL"))
                last_action = 2

        # HOLD
        else:
            if len(agent.inventory) > 0 and last_action == 0 :
                # last_action = 0
                reward = -1

            elif len(agent.inventory) > 0 and last_action == 1 :
                # agent.inventory.pop(0)
                bought_price = agent.inventory.pop(0)
                delta = data[t] -  last_position_price #bought_price
                reward = delta
                total_profit += delta
                history.append((data[t], "CLOSED_BUY"))
                history.append((data[t], "HOLD"))
                last_action = 0
                last_position_price = 0

            elif len(agent.inventory) > 0 and last_action == 2 :
                sold_price = agent.inventory.pop(0)
                delta = last_position_price - data[t] #sold_price - data[t]
                reward = delta
                total_profit += delta
                history.append((data[t], "CLOSED_SELL"))
                history.append((data[t], "HOLD"))
                last_action = 0
                last_position_price = 0

            else :
                agent.inventory.append(0)
                history.append((data[t], "HOLD"))
                last_action = 0
                last_position_price = 0

        if t == data_length - 1:
            if last_action == 1:
                bought_price = agent.inventory.pop(0)
                delta = data[t] - last_position_price
                total_profit += delta
                history.append((data[t], "CLOSED_BUY"))
            elif last_action == 2:
                sold_price = agent.inventory.pop(0)
                delta = last_position_price - data[t]
                total_profit += delta
                history.append((data[t], "CLOSED_SELL"))
            else:
                last_action = 0

            last_position_price = 0

        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            return total_profit, history







def open_position( action , last_action) :
    """opens a long or short position based on the action when the action is not similar to the last action
    """
    if action != last_action :
        if action == 1:
            return 1
        elif action == 2:
            return 2
        else:
            return 0


def close_position(agent , action , data , t , total_profit):
    """closes the position based on the action
    """
    if len(agent.inventory) > 0 and agent.inventory[-1] == action :
        sold_price = agent.inventory.pop(0)
        delta = sold_price - data[t]
        reward = delta
        total_profit += delta

    return  delta , reward , total_profit