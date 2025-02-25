import numpy as np
import random

class GaussMethod:
    def __init__(self):
        self.size = 0
        self.matrix = []
        self.k = 1
    def __input_from_keyboard(self):
        print("Введите размер матрицы")
        self.size = int(input())
        print("Введите строки матрицы")
        for i in range(self.size):
                self.matrix.append([int(j) for j in input().split()])
        print(self.matrix)

    def __input_from_file(self):
        matrix_filename = input("Введите имя файла: ")
        with open(matrix_filename, 'r') as f:
            self.size = int(f.readline())
            for i in range(self.size):
                self.matrix.append([int(j) for j in f.readline().strip().split()])
    def __input_random(self):
        print("Введите размер матрицы")
        self.size = int(input())
        for i in range(self.size):
            self.matrix.append([random.randint(1, 100) for i in range(self.size + 1)])

    def input_matrix(self):
        method = input("1 - ввод с клавиатуры, 2 - из файла, 3 - случайные значения: ")
        if method == "1":
            self.__input_from_keyboard()
        elif method == "2":
            self.__input_from_file()
        else:
            self.__input_random()

    def determinant(self):
        matrix = self.get_triangle()
        det = 1
        for i in range(self.size):
            det *= matrix[i][i]
        return det * self.k

    def get_triangle(self):
        matrix_triangle = [[self.matrix[i][j] for j in range(self.size+1)] for i in range(self.size)]
        for i in range(self.size):
            if matrix_triangle[i][i] == 0:
                for j in range(i + 1, self.size):
                    if matrix_triangle[j][i] != 0:
                        matrix_triangle[i], matrix_triangle[j] = matrix_triangle[j], matrix_triangle[i]
                        self.k *= -1
                        break
            if matrix_triangle[i][i] == 0:
                continue
            for j in range(i + 1, self.size):
                c = matrix_triangle[j][i] / matrix_triangle[i][i]
                for k in range(i, self.size + 1):
                    matrix_triangle[j][k] -= c * matrix_triangle[i][k]
        return matrix_triangle

    def solve(self):
        matrix = self.get_triangle()
        if any(all(matrix[i][j] == 0 for j in range(self.size)) for i in range(self.size)):
            print("Ошибка. Матрица не имеет решений или имеет больше одного решения.")
            return []
        matrix_ans = [0 for i in range(self.size)]
        for i in range(self.size - 1, -1, -1):
            matrix_ans[i] = matrix[i][self.size] / matrix[i][i]
            for j in range(i - 1, -1, -1):
                matrix[j][self.size] -= matrix[j][i] * matrix_ans[i]
        return matrix_ans

    def residual_vector(self):
        matrix_ans = self.solve()
        if not matrix_ans:
            print("Ошибка при решении")
            return []
        residual = [0 for i in range(self.size)]
        for i in range(self.size):
            residual[i] = sum([self.matrix[i][j] * matrix_ans[j] for j in range(self.size)])
            residual[i] -= self.matrix[i][self.size]
        return residual

    def get_a(self):
        return [[self.matrix[i][j] for j in range(self.size)] for i in range(self.size)]

    def get_b(self):
        return [self.matrix[i][self.size] for i in range(self.size)]
    def get_size(self):
        return self.size
def main():
    solver = GaussMethod()
    solver.input_matrix()
    det = solver.determinant()
    print(f"Определитель матрицы {det}")
    matrix_triange = solver.get_triangle()
    print(f"Треугольный вид матрицы {matrix_triange}")
    answer = solver.solve()
    print(f"Вектор неизвестных {answer}")
    res = solver.residual_vector()
    print(f"Вектор неувязок {res}")

    print("Сравнение с решением с библиотеки nympy")
    a = np.array(solver.get_a())
    b = np.array(solver.get_b())
    det_np = np.linalg.det(a)
    print(f"Определитель матрицы с помощью np {det_np}")
    print(f"Разница с нашим решением {abs(det_np-det)}")
    answer_np = np.linalg.solve(a, b)
    print(f"Вектор неизвестных с помощью np {answer_np}")
    print(f"Разница с нашим решением {[abs(answer_np[i]-answer[i]) for i in range(solver.get_size())]}")
if __name__ == "__main__":
    main()
