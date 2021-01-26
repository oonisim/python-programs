from typing import (
    List,
    Dict
)


def prime(num: int) -> List[int]:
    """Approach
    For i : i : 2, 3, ... until num
    all_numbers = range(1, 61, 1)
    From all numberse, keep removing all divisible by i
    """
    all_numbers: List[int] = list(range(2, num+1, 1))
    primes = []    # Result to accumulate the prime numbers
    while len(all_numbers) > 0:
        divider = all_numbers[0]
        primes.append(divider)

        # Remove all numbers divisible by the divider
        divisible: List[int] = []
        for to_remove in range(divider +1, num+1, 1):
            n = 2 * divider
            divisible.append(to_remove)

        print(f"TO remove is {divisible}")
        all_numbers = [stay for stay in all_numbers if stay not in divisible]
        print(all_numbers)
    return primes


if __name__ == "__main__":
    prime(60)