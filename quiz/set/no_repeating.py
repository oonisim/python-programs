# get a string with no repeating characters from a string.
# "acaacab" -> "b"


stack = []


def remove_repeats(element):
    if [element] == stack[-1:]:
        stack.pop()
    else:
        stack.append(element)


def main(elements):
    for element in elements:
        remove_repeats(element)

    print(stack)


main("abbaca")
