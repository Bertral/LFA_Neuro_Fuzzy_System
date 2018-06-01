from nfs.point import Point


class MF:
    def __init__(self, low: Point, mid: Point, high: Point):
        assert low.x < mid.x < high.x
        self.low = low
        self.mid = mid
        self.high = high

    def fuzzyfy(self, x: float):
        if x < self.low.x or x > self.high.x:
            return 0.0

        if x <= self.mid.x:
            return (x - self.low.x)/(self.mid.x - self.low.x)

        else:
            return (self.high.x - x)/(self.high.x - self.mid.x)

    def move(self, point: float, step: float, move_to: bool):
        sign = 1
        if not move_to:
            sign = -1

        dist_to_mid = point - self.mid.x
        self.mid.x += sign * step * dist_to_mid

        if point <= self.mid.x:
            self.low.x += sign * step * dist_to_mid

        else:
            self.high.x += sign * step * dist_to_mid
