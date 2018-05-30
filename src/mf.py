from src.point import Point


class MF:
    def __init__(self, low: Point, mid: Point, high: Point):
        assert low.x < mid.x < high.x
        self.low = low
        self.mid = mid
        self.high = high

    def fuzzyfy(self, x: Point):
        if x.x < self.low.x or x.x > self.high.x:
            return 0.0

        if x.x <= self.mid.x:
            return (x.x - self.low.x)/(self.mid.x - self.low.x)

        else:
            return (self.high.x - x.x)/(self.high.x - self.mid.x)
