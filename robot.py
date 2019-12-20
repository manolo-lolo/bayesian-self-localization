from typing import Dict, List

import numpy as np
from numpy.random import RandomState

RANDOM_SEED = 424242

MOVE_DIRECTION_TO_TEXT: Dict[int, str] = {-1: 'left', 1: 'right'}
ACTUAL_DIRECTION_TO_TEXT: Dict[int, str] = {-1: 'moving left', 0: 'not moving', 1: 'moving right'}
OBJECTS = ['X', 'O']
WRONG_MEASUREMENT: Dict[str, str] = {'X': 'O', 'O': 'X'}

try:
    from emoji import emojize
    ROBOT = emojize(':robot_face: ')
except ImportError:
    ROBOT = ' # '


class Robot:
    def __init__(self, world: List):
        self.__random_state: RandomState = RandomState(RANDOM_SEED)
        self.__world: np.ndarray = np.asarray(world)
        self.__world_length: int = self.__world.shape[0]
        self.__position: int = self.__random_state.randint(self.__world_length)
        self.__beliefs: np.ndarray = np.ones(self.__world_length) / self.__world_length
        self.__probability = self.__calc_probabilities()
        print(f'Starting at position {self.__position}.')
        self.__plot_beliefs_and_truth()

    def __plot_beliefs_and_truth(self) -> None:
        line_1 = 'Beliefs are:\t'
        line_2, line_3, line_4 = '\t\t', '\t\t', '\t\t'
        line_5 = 'Truth is:\t'
        for i in range(self.__world_length):
            line_1 += f'{int(round(self.__beliefs[i] * 100, 0))}%\t'
            line_2 += '###\t'
            line_3 += f' {i} \t'
            line_4 += f' {self.__world[i]} \t'
            line_5 += ROBOT + '\t' if i == self.__position else '\t'
        for line in ['', line_1, line_2, line_3, line_4, line_5, '']:
            print(line)
        if max(self.__beliefs) > 0.75:
            print(f'The robot is confident to be at position {self.__beliefs.argmax()}, which is '
                  + ('true.' if self.__beliefs.argmax() == self.__position else 'false.'))
        else:
            print('The robot is not confident about its position.')

    def __calc_probabilities(self) -> Dict[str, float]:
        probability = {
            'truth_O': (self.__world == 'O').sum() / self.__world_length,
            'truth_X': (self.__world == 'X').sum() / self.__world_length,
            'meas_O|truth_O': 0.95,
            'meas_X|truth_O': 0.05,
            'meas_O|truth_X': 0.05,
            'meas_X|truth_X': 0.95
        }
        probability['meas_O'] = probability['meas_O|truth_O'] * probability['truth_O'] \
                                + probability['meas_O|truth_X'] * probability['truth_X']
        probability['meas_X'] = 1 - probability['meas_O']
        probability['truth_O|meas_O'] = probability['meas_O|truth_O'] * probability['truth_O'] / probability['meas_O']
        probability['truth_X|meas_O'] = 1 - probability['truth_O|meas_O']
        probability['truth_O|meas_X'] = probability['meas_X|truth_O'] * probability['truth_O'] / probability['meas_X']
        probability['truth_X|meas_X'] = 1 - probability['truth_O|meas_X']
        return {key: round(value, 6) for key, value in probability.items()}

    def get_position(self) -> int:
        return self.__position

    def move(self) -> None:
        intended_direction: int = self.__random_state.choice([-1, 1])
        print(f'Try to move {MOVE_DIRECTION_TO_TEXT[intended_direction]}.')

        direction_modifier: int = self.__random_state.choice([1, -1, 0], p=[0.9, 0.05, 0.05])
        actual_direction: int = intended_direction * direction_modifier
        if direction_modifier == 1:
            print(f'Successfully moved {MOVE_DIRECTION_TO_TEXT[intended_direction]}.')
        else:
            print(f'Problem while moving. Actually {ACTUAL_DIRECTION_TO_TEXT[actual_direction]}.')

        self.__position = (self.__position + actual_direction) % self.__world_length
        shift = -1 * intended_direction
        shifted_belief = np.zeros_like(self.__beliefs)
        for i in range(self.__world_length):
            shifted_belief[i] = self.__beliefs[(i + shift) % self.__world_length] * 0.9
            shifted_belief[i] += self.__beliefs[(i - shift) % self.__world_length] * 0.05
            shifted_belief[i] += self.__beliefs[i] * 0.05
        self.__beliefs = shifted_belief

        print(f'Robot is at position {self.__position}')

    def measure(self) -> str:
        measurement = self.__world[self.get_position()]
        if self.__random_state.random() <= 0.95:
            print(f'Measurement successful. Measured {measurement}')
        else:
            print(f'Measurement failed. Actual object is {measurement}, but measured {WRONG_MEASUREMENT[measurement]}.')
            measurement = WRONG_MEASUREMENT[measurement]
        return measurement

    def update_beliefs(self, measurement: str) -> None:
        assert measurement in OBJECTS
        mask_O = (self.__world == 'O').astype(int)
        mask_X = (self.__world == 'X').astype(int)
        likelihood = mask_O * self.__probability[f'truth_O|meas_{measurement}'] / mask_O.sum() \
                     + mask_X * self.__probability[f'truth_X|meas_{measurement}'] / mask_X.sum()
        # self.__plot_likelihood(likelihood)
        norm = (likelihood * self.__beliefs).sum() + 1e-6
        updated_belief = np.maximum(likelihood * self.__beliefs / norm, 0.01)
        norm = updated_belief.sum() + 1e-6
        self.__beliefs = updated_belief / norm

    def step(self) -> None:
        self.move()
        measurement = self.measure()
        self.update_beliefs(measurement)
        self.__plot_beliefs_and_truth()


if __name__ == '__main__':
    world = ['X', 'X', 'X', 'O', 'X', 'O', 'X', 'O', 'X', 'X']
    r = Robot(world)
    while input() != 'q':
        r.step()
        print('Continue with [enter], quit with [q].')

