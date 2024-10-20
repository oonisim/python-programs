import sys
from typing import (
    List,
    Dict,
    Union
)


class MinStack:
    def __init__(self):
        self.stack: List[int] = []     # stack using list
        self.mins: List[int] = []      # To keep sorted version of stack
        self.min: int = -sys.maxsize

    def push(self, val: int) -> None:
        self.stack.append(val)
        self.mins = sorted(self.stack)

    def pop(self) -> int:
        if len(self.stack) == 0:
            raise RuntimeError("No element in stack")

        result: int = self.stack.pop()
        self.mins = sorted(self.stack)
        return result

    def getMin(self) -> int:
        if len(self.stack) == 0:
            raise RuntimeError("No element in stack")

        return self.mins[0]

    def top(self) -> int:
        """ Return the top of the stack"""
        if len(self.stack) == 0:
            raise RuntimeError("No element in stack")

        return self.stack[-1]


def main():
    keys: List[str] = ["MinStack","push","push","push","getMin","pop","top","getMin"]
    values: List[List[int]] = [[],[-2],[0],[-3],[],[],[],[]]

    assert len(keys) == len(values), "key/value pair mismatch"

    minStack: MinStack = MinStack()
    minStack.push(-2)
    minStack.push(0)
    minStack.push(-3)
    value = minStack.getMin() #  return -3
    assert value == -3
    minStack.pop()
    value = minStack.top()    #  return 0
    assert value == 0

    value = minStack.getMin() #  return -2
    assert value == -2

    print(minStack.stack)


if __name__ == "__main__":
    main()
