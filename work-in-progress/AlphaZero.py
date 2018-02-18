from aima3.games import (ConnectFour, RandomPlayer,
                         MCTSPlayer, QueryPlayer, Player,
                         MiniMaxPlayer, AlphaBetaPlayer,
                         AlphaBetaCutoffPlayer)
from collections import namedtuple
import numpy as np
import conx as cx
from keras import regularizers
import tensorflow as tf
from tqdm import tqdm
import random

def softmax_cross_entropy_with_logits(y_true, y_pred):
    """
    TensorFlow-based error function.
    """
    p = y_pred
    pi = y_true
    zero = tf.zeros(shape = tf.shape(pi), dtype=tf.float32)
    where = tf.equal(pi, zero)
    negatives = tf.fill(tf.shape(pi), -100.0)
    p = tf.where(where, negatives, p)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)
    return loss

## Building the network, layer blocks:

def add_conv_block(net, input_layer):
    cname = net.add(cx.Conv2DLayer("conv2d-%d",
                    filters=75,
                    kernel_size=(4,4),
                    padding='same',
                    use_bias=False,
                    activation='linear',
                    kernel_regularizer=regularizers.l2(0.0001)))
    bname = net.add(cx.BatchNormalizationLayer("batch-norm-%d", axis=1))
    lname = net.add(cx.LeakyReLULayer("leaky-relu-%d"))
    net.connect(input_layer, cname)
    net.connect(cname, bname)
    net.connect(bname, lname)
    return lname

def add_residual_block(net, input_layer):
    prev_layer = add_conv_block(net, input_layer)
    cname = net.add(cx.Conv2DLayer("conv2d-%d",
        filters=75,
        kernel_size=(4,4),
        padding='same',
        use_bias=False,
        activation='linear',
        kernel_regularizer=regularizers.l2(0.0001)))
    bname = net.add(cx.BatchNormalizationLayer("batch-norm-%d", axis=1))
    aname = net.add(cx.AddLayer("add-%d"))
    lname = net.add(cx.LeakyReLULayer("leaky-relu-%d"))
    net.connect(prev_layer, cname)
    net.connect(cname, bname)
    net.connect(input_layer, aname)
    net.connect(bname, aname)
    net.connect(aname, lname)
    return lname

def add_value_block(net, input_layer):
    l1 = net.add(cx.Conv2DLayer("conv2d-%d",
        filters=1,
        kernel_size=(1,1),
        padding='same',
        use_bias=False,
        activation='linear',
        kernel_regularizer=regularizers.l2(0.0001)))
    l2 = net.add(cx.BatchNormalizationLayer("batch-norm-%d", axis=1))
    l3 = net.add(cx.LeakyReLULayer("leaky-relu-%d"))
    l4 = net.add(cx.FlattenLayer("flatten-%d"))
    l5 = net.add(cx.Layer("dense-%d",
        20,
        use_bias=False,
        activation='linear',
        kernel_regularizer=regularizers.l2(0.0001)))
    l6 = net.add(cx.LeakyReLULayer("leaky-relu-%d"))
    l7 = net.add(cx.Layer('value_head',
        1,
        use_bias=False,
        activation='tanh',
        kernel_regularizer=regularizers.l2(0.0001)))
    net.connect(input_layer, l1)
    net.connect(l1, l2)
    net.connect(l2, l3)
    net.connect(l3, l4)
    net.connect(l4, l5)
    net.connect(l5, l6)
    net.connect(l6, l7)
    return l7

def add_policy_block(net, input_layer):
    l1 = net.add(cx.Conv2DLayer("conv2d-%d",
        filters=2,
        kernel_size=(1,1),
        padding='same',
        use_bias=False,
        activation='linear',
        kernel_regularizer = regularizers.l2(0.0001)))
    l2 = net.add(cx.BatchNormalizationLayer("batch-norm-%d", axis=1))
    l3 = net.add(cx.LeakyReLULayer("leaky-relu-%d"))
    l4 = net.add(cx.FlattenLayer("flatten-%d"))
    l5 = net.add(cx.Layer('policy_head',
                          42,
                          vshape=(6,7),
                          use_bias=False,
                          activation='linear',
                          kernel_regularizer=regularizers.l2(0.0001)))
    net.connect(input_layer, l1)
    net.connect(l1, l2)
    net.connect(l2, l3)
    net.connect(l3, l4)
    net.connect(l4, l5)
    return l5

def make_network(game, config, residuals=5, name="Residual CNN"):
    """
    Make a full network.

    Game is passed in to get the columns and rows.
    """
    net = cx.Network(name)
    net.add(cx.Layer("main_input", (game.v, game.h, 2),
                     colormap="Greys", minmax=(0,1)))
    out_layer = add_conv_block(net, "main_input")
    for i in range(residuals):
        out_layer = add_residual_block(net, out_layer)
    add_policy_block(net, out_layer)
    add_value_block(net, out_layer)
    net.compile(loss={'value_head': 'mean_squared_error',
                      'policy_head': softmax_cross_entropy_with_logits},
                optimizer="sgd",
                lr=config.LEARNING_RATE,
                momentum=config.LEARNING_RATE,
                loss_weights={'value_head': 0.5,
                              'policy_head': 0.5})
    for layer in net.layers:
        if layer.kind() == "hidden":
            layer.visible = False
    return net

class NNPlayer(Player):
    def __init__(self, name, net):
        super().__init__(name)
        self.net = net

    def set_game(self, game):
        """
        Get a mapping from game's (x,y) to array position.
        """
        self.game = game
        self.move2pos = {}
        self.pos2move = []
        position = 0
        for y in range(self.game.v, 0, -1):
            for x in range(1, self.game.h + 1):
                self.move2pos[(x,y)] = position
                self.pos2move.append((x,y))
                position += 1

    def get_predictions(self, state):
        """
        Given a state, give output of network on preferred
        actions. state.allowedActions removes impossible
        actions.

        Returns (value, probabilties, allowedActions)
        """
        board = np.array(self.state2array(state)) # 1 is my pieces, -1 other
        inputs = self.state2inputs(state)
        preds = self.net.propagate(inputs)
        value = preds[1][0]
        logits = np.array(preds[0])
        allowedActions = np.array([self.move2pos[act] for act in self.game.actions(state)])
        mask = np.ones(len(board), dtype=bool)
        mask[allowedActions] = False
        logits[mask] = -100
        #SOFTMAX
        odds = np.exp(logits)
        probs = odds / np.sum(odds)
        return (value, probs.tolist(), allowedActions.tolist())

    def get_action(self, state, turn):
        value, probabilities, moves = self.get_predictions(state)
        probs = np.array(probabilities)[moves]
        ## Probabilistically:
        ## pos = cx.choice(moves, probs)
        ## Best:
        pos = moves[cx.argmax(probs)]
        return self.pos2move[pos]

    def state2inputs(self, state):
        board = np.array(self.state2array(state)) # 1 is my pieces, -1 other
        currentplayer_position = np.zeros(len(board), dtype=np.int)
        currentplayer_position[board==1] = 1
        other_position = np.zeros(len(board), dtype=np.int)
        other_position[board==-1] = 1
        position = np.array(list(zip(currentplayer_position,other_position)))
        inputs = position.reshape((self.game.v, self.game.h, 2))
        return inputs

    def state2array(self, state):
        array = []
        to_move = self.game.to_move(state)
        for y in range(self.game.v, 0, -1):
            for x in range(1, self.game.h + 1):
                item = state.board.get((x, y), 0)
                if item != 0:
                    item = 1 if item == to_move else -1
                array.append(item)
        return array

class MCTSPlayerWithNetPolicy(MCTSPlayer):
    """
    A Monte Carlo Tree Search with policy function from
    neural network. Network will be set later to self.nnplayer.
    """
    def __init__(self, name, net, n_playout, c_puct, is_selfplay,
                 turn_to_play_deterministically,
                 *args, **kwargs):
        super().__init__(name,
                         *args,
                         n_playout=n_playout,
                         c_puct=c_puct,
                         is_selfplay=is_selfplay,
                         **kwargs)
        self.net = net
        self.turn_to_play_deterministically = turn_to_play_deterministically
        self.memory = []

    def set_game(self, game):
        super().set_game(game)
        self.nnplayer = NNPlayer(self.name + "-NNPlayer", self.net)
        self.nnplayer.set_game(game)

    def get_action(self, state, turn, *args, **kwargs):
        if turn >= self.turn_to_play_deterministically:
            self.temp = 1e-3
        else:
            self.temp = 0.1
        move, pi = super().get_action(state, turn, *args, return_prob=True, **kwargs)
        self.memory.append((self.nnplayer.state2inputs(state),
                            self.move_probs2all_probs(pi)))
        return move

    def move_probs2all_probs(self, move_probs):
        all_probs = np.zeros(len(self.nnplayer.state2array(game.initial)))
        for move in move_probs:
            all_probs[self.nnplayer.move2pos[move]] = move_probs[move]
        return all_probs.tolist()

    def policy(self, game, state):
        # these moves are positions:
        value, probs_all, moves = self.nnplayer.get_predictions(state)
        if len(moves) == 0:
            result = [], value
        else:
            probs = np.array(probs_all)[moves]
            moves = [self.nnplayer.pos2move[pos] for pos in moves]
            # we need to return probs and moves for game
            result = [(act, prob) for (act, prob) in list(zip(moves, probs))], value
        return result

class SelfPlayGame(ConnectFour):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = []

    def play_game(self, *players, **kwargs):
        results = super().play_game(*players, **kwargs)
        value = self.final_utility
        for player in players:
            if hasattr(player, "memory"):
                for state, probs in player.memory:
                    self.memory.append([state, [probs, [value]]])
                player.memory[:] = []
        return results

    def play_tournament(self, *players, **kwargs):
        self.memory = []
        return super().play_tournament(*players, **kwargs)

def train(config):
    """
    The AlphaZero training method.
    """
    global memory
    t0 = tqdm(total=config.CYCLES)
    for cycle in range(config.CYCLES):
        print("Cycle #%s..." % cycle)
        ########################################################
        # Phase 1: self-play, collect data
        ########################################################
        print("Self-play matches begin...")
        best_player1 = MCTSPlayerWithNetPolicy(
            "Best Player 1",
            best_net,
            config.N_PLAYOUT,
            config.C_PUCT,
            is_selfplay=True,
            turn_to_play_deterministically=config.TURN_TO_PLAY_DETERMINISTICALLY)
        best_player2 = MCTSPlayerWithNetPolicy(
            "Best Player 2",
            best_net,
            config.N_PLAYOUT,
            config.C_PUCT,
            is_selfplay=True,
            turn_to_play_deterministically=config.TURN_TO_PLAY_DETERMINISTICALLY)
        results = self_play_game.play_tournament(config.SELF_PLAY_MATCHES,
                                                 best_player1, best_player2)
        memory.extend(self_play_game.memory)
        ## Keep resonable size
        memory = memory[-config.MAX_MEMORY_SIZE:]
        print("Memory size is %s" % len(memory))
        if len(memory) >= config.MEMORY_SIZE:
            ########################################################
            # Phase 2: Train the current player with data collected:
            ########################################################
            print("Enough to train!")
            current_net.dataset.load(memory)
            current_net.train(config.TRAINING_EPOCHS_PER_CYCLE,
                                     batch_size=config.BATCH_SIZE, plot=False)
            ########################################################
            # Phase 3: Play current_player vs. best_player
            ########################################################
            ## If current is better thn best, move the current weights to best
            best_player = MCTSPlayerWithNetPolicy(
                "Best Player",
                best_net,
                config.N_PLAYOUT,
                config.C_PUCT,
                is_selfplay=False,
                turn_to_play_deterministically=1)
            current_player = MCTSPlayerWithNetPolicy(
                "Current Player",
                current_net,
                config.N_PLAYOUT,
                config.C_PUCT,
                is_selfplay=False,
                turn_to_play_deterministically=1)
            results = game.play_tournament(config.TOURNAMENT_MATCHES,
                                           best_player, current_player)
            print("Tournament results:", results)
            if results["Current Player"] > results["Best Player"] * config.BEST_SWAP_PERCENT:
                print("Current player won! swapping weights...")
                # copy the better weights to the best_player
                wts = current_net.get_weights()
                best_net.model.set_weights(wts)
                best_net.save_weights("BestWeights-%s.conx" % cycle)
            else:
                print("Best player remains best.")
        t0.update()
    t0.close()

Config = namedtuple("Config", [
    "CYCLES",
    "MEMORY_SIZE",
    "MAX_MEMORY_SIZE",
    "TRAINING_EPOCHS_PER_CYCLE",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "MOMENTUM",
    "SELF_PLAY_MATCHES",
    "TURN_TO_PLAY_DETERMINISTICALLY",
    "C_PUCT",
    "TOURNAMENT_MATCHES",
    "BEST_SWAP_PERCENT",
    "N_PLAYOUT",
])
config = Config(
    CYCLES=10, # number of cycles to run
    MEMORY_SIZE=5000, # min size of memory
    MAX_MEMORY_SIZE=30000, # min size of memory
    TRAINING_EPOCHS_PER_CYCLE=1, # training on current network
    BATCH_SIZE=256, # batch_size for NN training
    LEARNING_RATE=0.1, # SGD Learning rate
    MOMENTUM=0.9, # SGD Momentum
    SELF_PLAY_MATCHES=25, # matches to test yo' self per self-play round
    TURN_TO_PLAY_DETERMINISTICALLY=10, # for selfplay mode
    C_PUCT=1.0, # rely on priors (0, +Infinity]
    TOURNAMENT_MATCHES=20, # plays each player as first mover per match, so * 2
    BEST_SWAP_PERCENT=1.3, # you must be this much better than best
    N_PLAYOUT=50, # number of MCTS simulations
)

cx.clear_session()

self_play_game = SelfPlayGame()
game = ConnectFour()
memory = []
## The nets:
best_net = make_network(self_play_game, config)
current_net = make_network(self_play_game, config)

if __name__ == "__main__":
    train(config)
