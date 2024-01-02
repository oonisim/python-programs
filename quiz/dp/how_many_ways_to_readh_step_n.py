# Dynamic programming.
# How many ways to reach the step N using steps 1,2,..i.
steps = (1, 2, 3)


def get_to_n(n=10) -> int:
    # if we reach to the ground floor (n==0), the pass to here is one way to get to N.
    if n == 0:
        return 1
    if n < 0:
        return 0
    # Divide into sub problems. How many ways to get current step.
    sum([get_to_n(n - step) for step in steps])

