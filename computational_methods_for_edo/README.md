# Metódos Computacionais para a solução de equações diferenciais ordinárias.

In this project, I explored the use of computer-based iterative methods to solve differential equations. These equations are fundamental to a wide range of fields, including physics, engineering, and biology, and can be used to model a variety of phenomena. However, solving differential equations analytically can be difficult and time-consuming, making iterative methods an attractive alternative.

Implemented the iterative methods using Python and the Numpy library for numerical calculations. Numpy is a powerful tool for scientific computing, providing support for large, multi-dimensional arrays and matrices, as well as a wide range of mathematical functions to operate on these structures. However, the iterative nature of the methods meant that Numpy's vector optimization features could not be fully utilized, as each step of the iteration required the result of the previous one.

To address this issue, I utilized the Numba library, which allows for the compilation of Python code to native machine instructions. This can lead to significant performance improvements, particularly for computationally heavy tasks. By using Numba, I was able to achieve near-C performance for the iterative methods, making them much more efficient and scalable.

This work demonstrated my skills in numerical computing and optimization techniques in Python, as well as my ability to effectively utilize relevant libraries and tools. It also highlighted my understanding of differential equations and their role in modeling real-world phenomena, as well as my ability to apply iterative methods as a solution to solving these equations.
