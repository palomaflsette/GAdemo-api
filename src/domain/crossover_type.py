class CrossoverType:
    def __init__(self, one_point: bool, two_point: bool, uniform: bool):
        self.one_point = one_point
        self.two_point = two_point
        self.uniform = uniform