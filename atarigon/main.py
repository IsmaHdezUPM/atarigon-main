import argparse
import importlib.util
import inspect
import os
import random
import sys
import time
from typing import Type, List

from api import Goshi, Goban

MIN_BOARD_SIZE = 9

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

def run_game(
        *,
        goban: Goban,
        goshi: list[Goshi],
        shuffle=True
) -> dict[Goshi, int]:
    """Run a game of Go and return the scores for each player.

    :param goban: The board to play on.
    :param goshi: The players in the game.
    :param shuffle: Whether to shuffle the players before the game
        starts. Defauts to True.
    :return: A dictionary with the scores for each player.
    """

    # Randomize the order of the players
    goshi = goshi.copy()  # So we don't mess with the original list
    if shuffle:
        random.shuffle(goshi)

    kakunin = {g: 0 for g in goshi}  # The player's scores (確認)
    maketa = []  # The captured players (負けた)
    shoshinsha = []  # The players that doesn't know how to play (初心者)
    while len(goshi) > 1:
        player = goshi.pop(0)
        ten = player.decide(goban, kakunin[player])
        if ten is None:
            # If the player passes, it's added to the end of the list
            # and the next player is called
            goshi.append(player)
            continue

        # Ok, is a movement. If it is invalid, the player is
        # disqualified and the next player is called
        if not goban.seichō(ten, player):
            shoshinsha.append(player)
            kakunin[player] = 0
            continue

        # Stone is placed and captured players are removed from the game
        captured = goban.place_stone(ten, player)
        for captured_player in captured:
            # It maybe was an already captured player, so we check. If
            # not, the player score is incremented and the captured
            # player is removed from the game
            if captured_player in goshi:
                kakunin[player] += 1
                goshi.remove(captured_player)
                maketa.append(captured_player)

        # The player is added to the end of the list, waiting for its
        # next turn
        goshi.append(player)
    # Now we compute the scores based on the captured players and on
    # when and how they ended playing

    for player in goshi:
        # Para el jugador que sobrevive, el score es igual a la cantidad de jugadores eliminados
        # n-1
        kakunin[player] += len(maketa) + len(shoshinsha)
        player.update(goban, kakunin[player])
    for i, player in enumerate(reversed(maketa)):
        # Los jugadores que fueron eliminados tienen un score igual al numero de jugadores eliminados cuando ellos fueron eliminados
        kakunin[player] += i
        player.update(goban, kakunin[player])
    for player in shoshinsha:
        # Los jugadores que no saben jugar tienen un score igual a 0
        kakunin[player] = 0
        player.update(goban, kakunin[player])
    
    return kakunin


def find_subclasses(path: str, cls: Type[Goshi]):
    """Find all subclasses of a given class in a given path.

    :param path: The path to the directory where the modules are.
    :param cls: The class to find the subclasses of.
    :return: A list with all the subclasses of the given class in the
        given path.
    """
    goshi_classes = []

    sys.path.append(path)  # So we can import the modules from the path

    for filename in os.listdir(path):
        if filename.endswith('.py'):
            # Get the full module path name
            mod_name = filename[:-3]
            mod_path = os.path.join(path, filename)

            # Load the module
            spec = importlib.util.spec_from_file_location(mod_name, mod_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Fing all `Goshi` subclasses in the module
            for name, o in inspect.getmembers(module):
                if inspect.isclass(o) and issubclass(o, cls) and o is not cls:
                    goshi_classes.append(o)

    return goshi_classes


def main():
    parser = argparse.ArgumentParser(
        description='Play a game of Atari Go for N players'
    )
    parser.add_argument('--games', type=int, default=1, help='Number of games to play')
    parser.add_argument('--size', type=int, required=True, help='Size of the Go board')
    parser.add_argument('--agents', type=str, required=True, help='Directory to load agent files from')
    args = parser.parse_args()

    goshi_classes = find_subclasses(args.agents, Goshi)
    players = [clazz() for clazz in goshi_classes]

    # The player's scores (確認)
    kakunin: Type[Goshi, List[int]] = {player: [] for player in players}

    start_time = time.time()

    for game in range(args.games):
        results = run_game(
            goban=Goban(size=args.size, goshi=players),
            goshi=players,
        )
        for player, score in results.items():
            kakunin[player].append(score)
        # Print the score every 10 games
        if (game + 1) % 10 == 0:
            print(f"After {game + 1} games, scores are:")
            for player, score in kakunin.items():
                print(f"Player {player}: {sum(score)}")
    end_time = time.time()  # End the timer

    total_time = end_time - start_time
    time_per_game = total_time / args.games
    print(f"Total time for {args.games} games: {total_time} seconds")
    print(f"Average time per game: {time_per_game} seconds")
    # Leaderboard
    longest_name = max(max(len(str(p)) for p in players), len('Goshi'))
    print(f'{"Goshi":<{longest_name}} | {"Score":>6} | Individual Scores')
    for player, score in sorted(
            kakunin.items(), key=lambda x: sum(x[1]), reverse=True
    ):
        print(f'{str(player):<{longest_name}} | {sum(score):>6} | {score}')


if __name__ == '__main__':
    main()
