# bi_brainer.py
import sys
sys.path.append('aima-python')
import random
from games import *
from Monte_Carlo import MCTS
import time
def gomoku_evalX(state):
    ev = 0
    for col in range(1,17):
        for row in range(1,16):
            # favor center columns 
            if state.board.get((row,col)) == 'X':
                if (col == 8):
                    ev += (0.05)
                # avoid sides
                elif (col == 1 or col == 15):
                    ev-= (0.01)
            # disadvantage when enemy has center columns
            elif state.board.get((row, col)) == 'O':
                if (col == 8):
                    ev -= (0.05)
     # also check if enemy has three vertical pieces
    for col in range(1,16):
        enemy_pieces =0
        for row in range(1,17):
            # count how many enemy pieces are in this column
            if state.board.get((row, col)) == 'O':
                enemy_pieces += 1
                if enemy_pieces == 3:
                    ev -= 0.5
                elif enemy_pieces == 4:
                    ev -= 0.3
            else:
                enemy_pieces = 0  
    return ev

def bi_brainer_player(game, state, threshold=165, time_limit=0.1):
    available_moves = game.actions(state)
    if len(available_moves) <= threshold:
        print("Using Minimax. Available moves: ", len(available_moves))
        return alpha_beta_cutoff_search(state, game, d=2, eval_fn=gomoku_evalX)
    else:
        print("Using Monte Carlo Tree Search. Available moves: ", len(available_moves))
        mcts = MCTS()
        for i in range(int(time_limit * 100)):  
            mcts.do_rollout(game, state)
        best_state = mcts.choose(game, state)
        for move in available_moves:
            if game.result(state, move) == best_state:
                return move
        return random.choice(available_moves) 
def ab_cutoff_player(game, state):
    return alpha_beta_cutoff_search(state, game, d=2, eval_fn=gomoku_evalX)

def mcts_player(game, state, time_limit=0.1):
    available_moves = game.actions(state)
    print("Using Monte Carlo Tree Search. Available moves: ", len(available_moves))
    mcts = MCTS()
    for i in range(int(time_limit * 100)):  
        mcts.do_rollout(game, state)
    best_state = mcts.choose(game, state)
    for move in available_moves:
        if game.result(state, move) == best_state:
            return move
    return random.choice(available_moves) 
def main():
    n_iter=10
    # bi brain 
    nx= 0
    gm = Gomoku()
    start = time.time()
    for i in range(0,n_iter):
        if (gm.play_game(mcts_player, random_player)>0):
            nx+=1
        print("number of wins:", nx)
    end = time.time()
    print("Average time for MCTS player vs random_player in Gomoku:", (end-start)/n_iter)
    print("\nWin rate: ", nx/n_iter * 100, "%")
    # Minimax player
    nx= 0
    gm = Gomoku()
    start = time.time()
    for i in range(0,n_iter):
        if (gm.play_game(ab_cutoff_player, random_player)>0):
            nx+=1
        print("number of wins:", nx)
    end = time.time()
    print("Average time for Minimax player vs random_player in Gomoku:", (end-start)/n_iter)
    print("\nWin rate: ", nx/n_iter * 100, "%")

    # bi brain 
    nx= 0
    gm = Gomoku()
    start = time.time()
    for i in range(0,n_iter):
        if (gm.play_game(bi_brainer_player, random_player)>0):
            nx+=1
        print("number of wins:", nx)
    end = time.time()
    print("Average time for Bi_Brain vs random_player in Gomoku:", (end-start)/n_iter)
    print("\nWin rate: ", nx/n_iter * 100, "%")
    
if __name__ == "__main__":
    main()
