def find_duplicates(seq):
    duplicates = {}
    for element in seq:
        duplicates[element] = duplicates.get(element, 0) + 1

    return sum(1 for element in duplicates if duplicates[element] > 1)


print(find_duplicates([1, 1, 2, 5, 1, 2, 2, 6, 7, 9, 4]))
