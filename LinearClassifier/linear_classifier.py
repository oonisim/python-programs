def ThreePoints(coordinates):
    p1 = coordinates[0]
    p2 = coordinates[1]
    p3 = coordinates[2]

    first = p1[1:-1].split(',')
    print(first)
    x1 = int(first[0])
    y1 = int(first[1])
    second = p2[1:-1].split(',')
    x2 = int(second[0])
    y2 = int(second[1])
    thrid = p3[1:-1].split(',')
    x3 = int(thrid[0])
    y3 = int(thrid[1])

    X2 = x2 - x1
    Y2 = y2 - y1
    X3 = x3 - x1
    Y3 = y3 - y1

    w1 = -1 * Y2
    w2 = X2

    product = (w1 * X3) + (w2 * Y3)

    if product < 0:
        return "right"
    if product > 0:
        return "left"
    return "neither"

inp =["(0,0)", "(0,5)", "(0,2)"]
print(ThreePoints(inp))

# keep this function call here
#print(ThreePoints(input()))
