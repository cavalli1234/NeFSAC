import math


def sampson_distance(source_point, destination_point, model):
    return math.sqrt(squared_sampson_distance(source_point, destination_point, model))


def squared_sampson_distance(source_point, destination_point, model):
    x1 = source_point[0]
    y1 = source_point[1]
    x2 = destination_point[0]
    y2 = destination_point[1]

    e11 = model[0, 0]
    e12 = model[0, 1]
    e13 = model[0, 2]
    e21 = model[1, 0]
    e22 = model[1, 1]
    e23 = model[1, 2]
    e31 = model[2, 0]
    e32 = model[2, 1]
    e33 = model[2, 2]

    rxc = e11 * x2 + e21 * y2 + e31
    ryc = e12 * x2 + e22 * y2 + e32
    rwc = e13 * x2 + e23 * y2 + e33
    r = (x1 * rxc + y1 * ryc + rwc)
    rx = e11 * x1 + e12 * y1 + e13
    ry = e21 * x1 + e22 * y1 + e23

    return r * r / (rxc * rxc + ryc * ryc + rx * rx + ry * ry)
