"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time

class model :
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, d_model, enc_voc_size, dec_voc_size, max_len, ffn_hidden, n_heads, n_layers, drop_prob, device):
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.d_model = d_model
        self.enc_voc_size = enc_voc_size
        self.dec_voc_size = dec_voc_size
        self.max_len = max_len
        self.ffn_hidden = ffn_hidden
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.device = device
        self.model = Transformer(src_pad_idx=src_pad_idx,
                                 trg_pad_idx=trg_pad_idx,
                                 trg_sos_idx=trg_sos_idx,
                                 d_model=d_model,
                                 enc_voc_size=enc_voc_size,
                                 dec_voc_size=dec_voc_size,
                                 max_len=max_len,
                                 ffn_hidden=ffn_hidden,
                                 n_head=n_heads,
                                 n_layers=n_layers,
                                 drop_prob=drop_prob,
                                 device=device).to(device)
        self.optimizer = Adam(params=model.parameters(),
                              lr=init_lr,
                              weight_decay=weight_decay,
                              eps=adam_eps)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                verbose=True,
                                                                factor=factor,
                                                                patience=patience)
        self.criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

    def train_with_Qlearning(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        target_batch = []
        for i in range(0, batch_size):
            target = rewards[i]
            if not dones[i]:
                target = (rewards[i] + self.gamma * np.amax(self.model.predict(next_states[i])[0]))
            target_f = self.model.predict(states[i])
            target_f[0][actions[i]] = target
            target_batch.append(target_f)
        self.model.update(states, np.array(target_batch), self.learning_rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, model, iterator, optimizer, criterion, clip):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            optimizer.zero_grad()
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            epoch_loss += loss.item()
            print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

        return epoch_loss / len(iterator)


    def run(self, total_epoch, best_loss):
        train_losses, test_losses, bleus = [], [], []
        for step in range(total_epoch):
            start_time = time.time()
            train_loss = self.train(self.model, train_iter, optimizer, criterion, clip)
            valid_loss, bleu = self.evaluate(self.model, valid_iter, criterion)
            end_time = time.time()

            if step > warmup:
                scheduler.step(valid_loss)

            train_losses.append(train_loss)
            test_losses.append(valid_loss)
            bleus.append(bleu)
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

            f = open('result/train_loss.txt', 'w')
            f.write(str(train_losses))
            f.close()

            f = open('result/bleu.txt', 'w')
            f.write(str(bleus))
            f.close()

            f = open('result/test_loss.txt', 'w')
            f.write(str(test_losses))
            f.close()

            print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

def _init_model(src_pad_idx, trg_pad_idx, trg_sos_idx, d_model, enc_voc_size, dec_voc_size, max_len, ffn_hidden, n_heads, n_layers, drop_prob, device):
    return Transformer(src_pad_idx=src_pad_idx,
                       trg_pad_idx=trg_pad_idx,
                       trg_sos_idx=trg_sos_idx,
                       d_model=d_model,
                       enc_voc_size=enc_voc_size,
                       dec_voc_size=dec_voc_size,
                       max_len=max_len,
                       ffn_hidden=ffn_hidden,
                       n_head=n_heads,
                       n_layers=n_layers,
                       drop_prob=drop_prob,
                       device=device).to(device)
def add_optimizer(model, init_lr, weight_decay, adam_eps):
    return Adam(params=model.parameters(),
                lr=init_lr,
                weight_decay=weight_decay,
                eps=adam_eps)
def add_scheduler(optimizer, factor, patience):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                verbose=True,
                                                factor=factor,
                                                patience=patience)
def add_criterion(src_pad_idx):
    return nn.CrossEntropyLoss(ignore_index=src_pad_idx)



model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
