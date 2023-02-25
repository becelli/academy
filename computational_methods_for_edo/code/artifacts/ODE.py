class ODE_Solver:
    def __init__(self, f, a, b, y0):
        super().__init__()
        self.f = f
        self.a = a
        self.b = b
        self.y0 = y0
