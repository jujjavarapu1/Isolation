import pdb
"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def move_is_legal(move, board):
    row, col = move
    return 0 <= row < len(board) and \
           0 <= col < len(board[0]) and \
           board[row][col] == 0

def get_legal_moves(move, board):
    #Legal moves from a move.
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2),  (1, 2), (2, -1),  (2, 1)]
    (r,c) = move
    valid_moves = [(r+dr,c+dc) for dr, dc in directions if move_is_legal((r+dr, c+dc), board)]
    return valid_moves

def box_value(game, box):
    (x,y) = box
    value = 1000
    #Box Values
    if (x-2) >= 0:
        if (y-1) >= 0 :
            value += 1
        if (y+1) < game.width:
            value += 1
    if (x+2) < game.height:
        if (y-1) >= 0 :
            value += 1
        if (y+1) < game.width:
            value += 1
    if (y-2) >= 0:
        if (x-1) >= 0 :
            value += 1
        if (x+1) < game.height:
            value += 1
    if (y+2) < game.width:
        if (x-1) >= 0 :
            value += 1
        if (x+1) < game.height:
            value += 1
    return value

def dfs_board(move, board):
    (x, y) = move
    if board[x][y]:
        return 0
    else:
        board[x][y] = 1
        legal_moves = get_legal_moves(move,board)
        dfs_moves_count = [0]
        for m in legal_moves:
            dfs_moves_count.append(dfs_board(m, board))
        return max(dfs_moves_count) + 1


def heuristic1(game, player):

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    common_moves = list(set(own_moves) & set(opp_moves))
    #If one of the move can be occupied by opposition
    if (len(common_moves) != 0) and (game.inactive_player == player):
        return float(len(own_moves) - len(opp_moves) - 1)
    else:
        return float(len(own_moves) - len(opp_moves))

def heuristic2(game, player):

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    own_move_values = 0
    opp_move_values = 0

    for move in own_moves:
        own_move_values += box_value(game, move)
    for move in opp_moves:
        opp_move_values += box_value(game, move)
    return float(own_move_values - opp_move_values)

def heuristic3(game, player):

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    own_dfs_moves_count = [0]
    # From each move to how many move we can transverse.
    for move in own_moves:
        board = game.copy_board()
        own_dfs_moves_count.append(dfs_board(move, board))

    opp_dfs_moves_count = [0]
    for move in opp_moves:
        board = game.copy_board()
        opp_dfs_moves_count.append(dfs_board(move, board))

    return float(1000*(len(own_moves) - len(opp_moves)) + sum(own_dfs_moves_count))

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    if (len(game.get_blank_spaces()) >= (game.width * game.height)/2):
        #Using heuristic1 in intial part of game.
        return heuristic1(game, player)
    else:
        #Using heuristic3 in middle part of game.
        return heuristic3(game, player)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=15.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        best_move = None
        best_score = 0

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if len(legal_moves) == 0:
                return (-1,-1)
            if self.iterative:
                i = 1
                for depth_level in range(10):
                    scores = []
                    for move in legal_moves:
                        game_copy = game.forecast_move(move)
                        if self.method == "alphabeta":
                            score, _ = self.alphabeta(game_copy, depth_level, maximizing_player= False)
                        else:
                            score, _ = self.minimax(game_copy, depth_level, maximizing_player= False)
                        scores.append((score, move))
                    best_score, best_move = max(scores)
                return best_move
            else:
                scores = []
                for move in legal_moves:
                    game_copy = game.forecast_move(move)
                    score, _ = self.minimax(game_copy, self.search_depth - 1, maximizing_player= False)
                    scores.append((score, move))
                best_score, best_move = max(scores)
                return best_move

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return best_move

        # Return the best move from the last completed search iteration

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < (self.TIMER_THRESHOLD):
            raise Timeout()

        if depth == 0:
            return (self.score(game, self), game.get_player_location(game.inactive_player))
        else:
            scores = []
            legal_moves = game.get_legal_moves()
            for move in legal_moves:
                game_copy = game.forecast_move(move)
                score, _ = self.minimax(game_copy, depth-1, maximizing_player= not maximizing_player)
                scores.append((score,move))

            if len(scores) == 0:
                return (self.score(game, self), (-1,-1))

            if maximizing_player:
                return max(scores)
            else:
                return min(scores)


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < (self.TIMER_THRESHOLD):
            raise Timeout()

        if depth == 0:
            return self.score(game, self), game.get_player_location(game.inactive_player)
        else:
            if maximizing_player:
                best_score , best_move = float("-inf"), (-1,-1)
            else:
                best_score , best_move = float("inf"), (-1,-1)
            legal_moves = game.get_legal_moves()
            for move in legal_moves:
                game_copy = game.forecast_move(move)
                score, _ = self.alphabeta(game_copy, depth-1, alpha=alpha, beta=beta, maximizing_player= not maximizing_player)
                if maximizing_player:
                    if score < beta:
                        best_score, best_move = max([(best_score, best_move), (score, move)])
                        alpha = best_score
                    else:
                        best_score, best_move = score, move
                        break
                else:
                    if alpha < score:
                        best_score, best_move = min([(best_score, best_move), (score, move)])
                        beta = best_score
                    else:
                        best_score, best_move = score, move
                        break

            if len(legal_moves) == 0:
                return (self.score(game, self), (-1,-1))
            return best_score, best_move
