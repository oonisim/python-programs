#!/usr/bin/env python
"""
OBJECTIVE:
Implement a robot which can move on a board of size n x m.
"""
from typing import (
    Optional,
    Union,
    Tuple,
    List,
    Dict,
    TypedDict,
    Final
)
import logging
import numpy as np
import mathematics
from constant import (
    DIRECTIONS,
    DIRECTION_TO_MOVE,
    MOVE_TO_DIRECTION,
    MOVES,
    PLACE,
    REPORT,
    MOVE,
    LEFT,
    RIGHT,
    COMMAND_ACTIONS
)
from area import Board


class State(TypedDict):
    """Robot state to isolate/struct its state machine"""
    location: List[int]
    direction: str


class Command(TypedDict):
    """Robot command"""
    action: str
    state: State


class Robot:
    """Robot class that executes the robot commands while staying on the board.
    The board is aligned with Cartesian coordinate system.
    Its bottom left corner has coordinate (0,0) and top right (n-1, m-1).
    Robot cannot move off the board and ignores a command forcing it to do it.
    A newly created robot must be placed into location x=0 y=0 and face NORTH.

    The commands that robot understands:
    PLACE x y direction     -- re-places robot into the location set the direction
    MOVE                    -- moves robot one step in the current direction
    LEFT                    -- turns robot 90 degrees anticlockwise
    RIGHT                   -- turns robot 90 degrees clockwise
    REPORT                  -- outputs current state into STDOUT

    APPROACH:
    location is represented as a vector (x, y). move is represented as vector (dx, dy).
    move is either (0, 1): NORTH, (1, 0): EAST, (-1, 0): WEST, or (0, -1): SOUTH.

    At MOVE command, the current location is updated by np.add(location, move) if the
    destination is still on the self._board. np.zero(2) <= location <= np.array([n-1, m-1)]).

    LEFT/RIGHT is a rotation of the current move. New move is set as rotation.dot(move).
    ```
        rotation = np.array([
            [np.cos(radian), np.sin(radian)],
            [-np.sin(radian), np.cos(radian)]
        ])
    ```
    """
    # --------------------------------------------------------------------------------
    # Class initialization
    # --------------------------------------------------------------------------------
    ROTATE_LEFT_MATRIX = mathematics.rotation_matrix(-90)
    ROTATION_LEFT_MATRIX = mathematics.rotation_matrix(90)

    # --------------------------------------------------------------------------------
    # Instance initialization
    # --------------------------------------------------------------------------------
    def __init__(self, board: Board, state: State, log_level=logging.ERROR) -> None:
        """Initialize Robot class instance"""
        assert state['direction'] in DIRECTIONS, \
            "direction is incorrect {}".format(state['direction'])
        assert self._board.contains(state['location']), \
            "location {} outside the board ({})".format(state['location'], self._board.shape)

        self._board: Board = board      # Board where the robot is placed
        self._state: State = state      # Robot state

        logging.basicConfig()
        self._log_level = log_level
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)

    # --------------------------------------------------------------------------------
    # Command functions.
    # Keep it stateless and manage the state machine in one place (execute()).
    # However, not using @staticmethod as per Google Style.
    # --------------------------------------------------------------------------------
    def _place(self, state: State) -> Optional[State]:
        """Set the new state to the robot
        Args:
            state: new state
        Returns:
            new state after the command if it was valid or None
        """
        assert 'direction' in state and 'location' in state, \
            "_place(): invalid state".format(state)

        if state['direction'] in DIRECTIONS and self._board.contains(state['location']):
            return state
        else:
            self._logger.debug("_place(): Not setting to the new state {}".format(state))
            return None

    def _report(self, state: State) -> State:
        """Report the current location and direction"""
        formatting: str = "X: {} Y: {} direction: {}"
        print(formatting.format(
            state['location'][0],
            state['location'][1],
            state['direction']
        ))
        return state

    def _move(self, state) -> State:
        """move a step in the current direction if the destination is in the board
        Args:
            state: state of the robot
        Returns:
            New robot state
        """
        destination = np.add(state['location'], DIRECTION_TO_MOVE[state['direction']])
        if self._board.contains(destination):
            self._logger.debug("_move(): moved from {} to destination {}".format(
                state['location'], destination
            ))
            state['location'] = destination
        else:
            self._logger.debug("_move(): did not move from {} to destination {}".format(
                state['location'], destination
            ))

        return state

    def _left(self, state: State) -> State:
        """Turn the robot to the left
        Args:
            state: Current state of the robot
        Returns:
            Robot state after the command
        """
        vector = DIRECTION_TO_MOVE[state['direction']]
        rotated = Robot.ROTATE_LEFT_MATRIX.dot(vector).astype(int)
        self._logger.debug("_left() pre-step {} post step {}".format(vector, rotated))

        state['direction'] = MOVE_TO_DIRECTION[rotated]
        return state

    def _right(self, state: State) -> State:
        """Turn the robot to the right
        Args:
            state: Current state of the robot
        Returns:
            Robot state after the command
        """
        vector = DIRECTION_TO_MOVE[state['direction']]
        rotated = Robot.ROTATION_LEFT_MATRIX.dot(vector).astype(int)
        self._logger.debug("_right() pre-step {} post step {}".format(vector, rotated))

        state['direction'] = MOVE_TO_DIRECTION[rotated]
        return state

    # --------------------------------------------------------------------------------
    # I/F
    # --------------------------------------------------------------------------------
    def execute(self, command: Command) -> State:
        """Execute a command if it is a valid command, or ignore
        Args:
            command: Command
        Returns:
            Robot state after the command if executed.
        """
        assert command and 'action' in command and command['action'].upper() in COMMAND_ACTIONS
        command['action'] = command['action'].upper()

        self._logger.debug("execute: action {}: command {} current state {}".format(
            command['action'], command, self._state
        ))

        if command['action'] == PLACE:
            assert 'state' in command
            posterior = Robot._place(state=command['state'])
            self._state = posterior if posterior else self._state
        elif command['action'] == LEFT:
            self._state = Robot._left(state=self._state)
        elif command['action'] == RIGHT:
            self._state = Robot._right(state=self._state)
        elif command['action'] == MOVE:
            self._state = Robot._move(state=self._state)
        elif command['action'] == REPORT:
            Robot._report(state=self._state)
        else:
            self._logger.error("execute: unknown command {}".format(command))
            pass

        self._logger.debug("execute: state after the command is {}".format(self._state))

        return self._state
