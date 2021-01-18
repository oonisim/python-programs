# Task
```
Design and implement a solution for a Robot which can move on a board of size n x m.

Requirements
------------

1. Robot is always facing one of the following directions: NORTH, EAST,SOUTH or WEST.
2. Robot can turn left, right, and move one step at a time.
3. Robot cannot move off the board and ignores a command forcing it to do it.
4. Robot can report its current position on the board and direction.
5. A newly created robot must be placed into location x=0 y=0 and face NORTH.
6. The board is aligned with Cartesian coordinate system. Its bottom left corner has coordinate (0,0) and top right (n-1, m-1).

The solution will receive commands as a text file and outputs robot's reports into STDOUT. Below are the only commands that robot understands:

PLACE x y direction     -- places robot into specified location on a board and set initial direction
MOVE                    -- moves robot one step in the current direction
LEFT                    -- turns robot 90 degrees anticlockwise
RIGTH                   -- turns robot 90 degrees clockwise
REPORT                  -- outputs current state into STDOUT


For example, the following input file:
PLACE 0 0 NORTH
MOVE
REPORT
PLACE 0 0 NORTH
LEFT
REPORT
PLACE 1 2 EAST
MOVE
MOVE
LEFT
MOVE
REPORT

should produce an output similar to:
X: 0 Y: 1 Direction: NORTH
X: 0 Y: 0 Direction: WEST
X: 3 Y: 3 Direction: NORTH
```

# How to run
## System requirements
1. Python 3.8.x which support the assignment expression
(https://www.python.org/dev/peps/pep-0572/)
2. numpy (see requirements.txt)
3. pytest to run test scripts (see requirements.txt)

### Run program
In the root folder:
1. Run ```pip3 install -r requirements.txt```.
2. Run ```chmod u+x ./run.sh && ./run.sh```.

### Run test
To run the pytest:
1. ```cd package```.
2. Run ```pytest -ra --verbose```.


pylint has a known open issue of reporting "relative import beyond top-level package".
[pylint false positive attempted relative import beyond top-level package](https://github.com/flycheck/flycheck/issues/1758)


# Directory structure
```
<root>
├── README
├── requirements.txt                # Python requirement packages
├── run.sh                          # Run the program
├── package                         # Source directory
│  ├── __init__.py
│  ├── main.py
│  ├── constant.py                  # Constant declarations
│  ├── mathematics.py               # math utility e.g. vector rotation
│  ├── area.py                      # Board class (to avoid the standard package name)
│  ├── robot.py                     # Robot class
│  ├── operator.py                  # Operator class
│  ├── test_00_config.py            # Pytest common configuration
│  ├── test_010_board.py            # Board class test case for Board creation
│  ├── test_011_board.py            # Board class test case for Board boundary checks
│  ├── test_020_robot_creation.py   # Robot class test case for Robot creation
│  ├── test_030_robot_operation.py  # Robot class test case for Robot control operations
│  ├── test_040_operator_handles_invalid_path.py
│  ├── test_041_config.py           # Pytest configuration for test_041 test case.
│  ├── test_041_operator_read_command.py    # Operator class test case for reading commands
│  └── data                     # Data directory
│     ├── command.txt
│     ├── commands.txt
│     ├── test_041_empty.txt
│     ├── test_041_invalid_commands.txt
│     └── test_050_commands.txt
├── run_test.sh                 # Not used for now
└── test                        # Test directory (Not used for now)
```

# Design

## Board
The responsibility of the Board class is to provide the board where a robot operates and its utilities to check the
board boundary to tell if a location is within its boundaries.

## Operator
The responsibility of the Operator class is to provide a communication I/F to a robot.

* Create a robot and sending commands to a robot to operate in a board.
* Handle a command source to cope with different data sources.

Each operator can operate on its own board and can accept any command source as a Python generator.
Different command source e.g. HTTP endpoint, message queue, etc would be possible.

To make a robot as a remote device e.g. IoT device, Operator is the remote communication I/F e.g. MQTT such as AWS IoT.
Operator will be managing the shadow state of the remote robot and manage the connection reties for unreliable communiation.

To orchestrate multiple robots, Operator.direct() can be used to synchronize the robots. Currently it is a generator which
runs on a single thread and blocks for a message sent to to the operator via the generator send() (or iterable next()) method.
By invoking generator's close(), an operator and its robot can be terminated.


## Robot
The responsibility of the Robot class is to execute the commands from its operator. Operator and Robot has one-to-one relationship.

# Unit testing

Using pytest to test each class. See test.doc for the details on test conditions and cases.

