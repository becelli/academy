from cmath import isclose
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn")


class Plotter2D:
    def plot_function(self, x, y, l: str = "Função", style="--"):
        if x is not None and y is not None:
            plt.plot(x, y, label=l, linestyle=style)

    def plot_bar(self, value, l: str = "?"):
        plt.bar(l, value)

    def plot_approximations(self, methods=[], F: callable = None) -> None:
        if F is None:
            self._plot_only_numerical_method(methods)
        else:
            self._setup_4_subplots()

            # Cartesian plot of the function and approximations.
            plt.subplot(2, 2, 1)
            self._plot_cartesian(methods, F)

            # Sum of the errors (log scale)
            plt.subplot(2, 2, 2)
            self._plot_total_error(methods, F)

            # Percentual Relative Error at each point.
            plt.subplot(2, 2, 3)
            self._plot_relative_error(methods, F)

            # Relative error at each point
            plt.subplot(2, 2, 4)
            self._plot_error(methods, F)

        plt.show()

    def _plot_only_numerical_method(self, methods=[]):
        self._setup_1_subplot()
        for (x, y, label) in methods:
            self.plot_function(x, y, label)
        plt.legend()
        plt.show()

    def _plot_cartesian(self, methods, f):
        plt.title("Gráfico das aproximações")
        for (x, y, label) in methods:
            self.plot_function(x, y, label)
        self.plot_function(x, f(x), "Solução", style="-")
        plt.legend()

    def _plot_total_error(self, methods, f):
        plt.title("Erro global")
        plt.yscale("log")
        for (x, y, label) in methods:
            error = np.sum(np.abs(y - f(x)))
            self.plot_bar(error, label)

    def _plot_relative_error(self, methods, f):
        plt.title("Erro relativo percentual")
        for (x, y, label) in methods:
            error = np.zeros(x.shape)
            for i in range(x.shape[0]):
                if isclose(y[i], 0):
                    error[i] = 0
                else:
                    error[i] = np.abs(y[i] - f(x[i])) / np.abs(y[i])
            self.plot_function(x, error, label)
        plt.legend()

    def _plot_error(self, methods, f):
        plt.title("Erros relativos")
        plt.yscale("log")
        for (x, y, label) in methods:
            e = np.abs(y - f(x))
            self.plot_function(x, e, label)
        plt.legend()

    def _setup_4_subplots(self):
        plt.rcParams["figure.figsize"] = [16, 16]
        plt.rcParams["figure.dpi"] = 64
        plt.subplots(2, 2)

    def _setup_1_subplot(self):
        plt.rcParams["figure.figsize"] = [8, 8]
        plt.rcParams["figure.dpi"] = 64
        plt.subplot(1, 1, 1)
