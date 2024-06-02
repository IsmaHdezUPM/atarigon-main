import random
from typing import Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque, namedtuple
from api import Goban, Goshi, Ten

DIMENSION = 19
EPSILON = 0.3
SAMPLE_SIZE = 16
MEMORY_SIZE = 200
BETTA = 0.9

class DeepQAgent(Goshi):
    def __init__(self):
        super().__init__(f'David')
        self.model = self._init_model()

        # Load the model weights
        try:
            self.model.load_weights(f'./deepq_agents/DeepQ_{DIMENSION}x{DIMENSION}_david.weights.h5')
        except:
            print("No weights found, creating new file...")
            self.model.save_weights(f'./deepq_agents/DeepQ_{DIMENSION}x{DIMENSION}_david.weights.h5')
        
        self.last_state = None
        self.last_action = None
        self.experience_replay = deque(maxlen=MEMORY_SIZE)
        self.last_kakunin = 0

    def decide(self, goban: 'Goban', kakunin) -> Optional[Ten]:
        # If this is the first move, we wait for the next state to be available before storing
        if self.last_state is not None:
            # Store the experience in the experience replay
            # Score is the difference between the current kakunin and the last kakunin. This means that if we are disqualified we automatically get a score of -1, and that captures are treated as a small reward. 
            self._store_experience(self._encode_goban(goban), kakunin-self.last_kakunin)
        
        action = self._get_next_action(goban)
        
        # We will be generally working with a simpler internal representation.
        self.last_state = self._encode_goban(goban)
        self.last_kakunin = kakunin
        self.last_action = action
        return action
    
    def update(self, goban: 'Goban', kakunin) -> Optional[Ten]:
        # This is called at the end of each episode
        
        # We store the last experience. We will set next state to none as it is not needed for final states and it will also help us identify them
        self._store_experience(None, kakunin)
        if len(self.experience_replay) > SAMPLE_SIZE:
            self._train_model()
        self.model.save_weights(f'./deepq_agents/DeepQ_{DIMENSION}x{DIMENSION}_david.weights.h5')
        # After the update, we will start a new game, so we reset internal counters
        self.last_state = None
        self.last_action = None
        self.last_kakunin = 0

    def _move_like_a_ninja(self, goban):
        #Applies ninja's logic to decide the next move (randomly chooses an empty position on the board)
        empty_positions = [
            Ten(row, col)
            for row in range(len(goban.ban))
            for col in range(len(goban.ban[row]))
            if goban.ban[row][col] is None
        ]
        # Chooses a random valid empty position
        random.shuffle(empty_positions)
        for ten in empty_positions:
            if goban.ban[ten.row][ten.col] is None:
                return ten
        else:
            return None

    def _encode_goban(self, goban):
        # Get the dimensions of ban
        rows = len(goban.ban)
        cols = len(goban.ban[0]) if rows > 0 else 0

        # Initialize a 3D numpy array of zeros
        one_hot = np.zeros((rows, cols, 3))

        # Iterate over each position in goban.ban
        for i in range(rows):
            for j in range(cols):
                if goban.ban[i][j] is None: #Empty positions
                    one_hot[i, j, 0] = 1
                elif goban.ban[i][j] == self: #Self positions
                    one_hot[i, j, 1] = 1
                else:  # Any other object is a rival
                    one_hot[i, j, 2] = 1
        return one_hot
    
    def _print_goban(self, goban):
        # Iterate over each position in goban.ban
        # For debugging purposes, but I must say it looks cool
        print("Goban:")
        for row in goban.ban:
            for position in row:
                if position is None:
                    print("[ ]", end="")
                elif position == self:  
                    print("[○]", end="")
                else: 
                    print("[●]", end="")
            print() 

    def _store_experience(self, next_state, reward):
        experience={
            'state': self.last_state,
            'action': self.last_action,
            'reward': reward,
            'next_state': next_state
        }
        # Queue-like behavior, remove the oldest experience
        if len(self.experience_replay) >= MEMORY_SIZE:
            self.experience_replay.pop()
        self.experience_replay.appendleft(experience)

    def _init_model(self):
        model = tf.keras.Sequential([
            layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(DIMENSION, DIMENSION, 3)),
            layers.Flatten(),
            layers.Dense(512, activation='softmax'),
            layers.Dense(DIMENSION*DIMENSION, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def _get_next_action(self, goban):
        # Implements epsilon-greedy policy
        if np.random.rand() < EPSILON:
            return self._move_like_a_ninja(goban)
        else:
            return self._get_best_action(goban)
        
    def _get_best_action(self, goban):
        state = self._encode_goban(goban)
        state = np.expand_dims(state, axis=0)
        # Predict using model
        action_values = self.model.predict(state, verbose=0)

        # Get the position of the maximum value, ignoring invalid moves
        row, col = np.unravel_index(np.argmax(np.where(state[:,:,:,0].flatten() == 0, -np.inf, action_values)), (DIMENSION,DIMENSION))
        return Ten(row, col)

    def _train_model(self):
        # Sample a random batch from the experience replay
        batch = random.sample(self.experience_replay, SAMPLE_SIZE)
        states = []
        targets = []
        for experience in batch:
            # Generate target based on observed reward, then train the model
            state = np.expand_dims(experience['state'], axis=0)
            action = experience['action']
            reward = experience['reward']
            next_state = experience['next_state']
            # Initialize the target to the model's prediction
            target = self.model.predict(state, verbose=0)
            action_index = action[0] * DIMENSION + action[1]
            
            if next_state is None:  # If the state is final
                target[0, action_index] = reward
            else:  # If the state is not final
                # Predict the future discounted reward
                next_state = np.expand_dims(next_state, axis=0)
                # Also prevent it from considering invalid moves
                future_reward = np.amax(np.where(next_state[:,:,:,0].flatten() == 0, -np.inf, self.model.predict(next_state, verbose=0)[0]))
                target[0, action_index] = reward + BETTA * future_reward
            
            states.append(state[0])
            targets.append(target[0])

        # Convert lists to numpy arrays
        states = np.array(states)
        targets = np.array(targets)

        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0)

    def _print_state(self, state):
        # For debugging purposes
        for row in state:
            for column in row:
                for position in column:
                    if np.array_equal(position, [1, 0, 0]):  # Empty position
                        print("[ ]", end="")
                    elif np.array_equal(position, [0, 1, 0]):  # Self
                        print("[○]", end="")
                    elif np.array_equal(position, [0, 0, 1]):  # Opponent
                        print("[●]", end="")
                print()

    def _print_target(self, target):
        # For debugging purposes
        target_grid = target.reshape((DIMENSION, DIMENSION))

        for row in target_grid:
            for value in row:
                print(f"{value:.2f}", end=" ")
            print()