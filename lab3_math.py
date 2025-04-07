import math

def left_rectangle(f, a, b, n):
    h = (b - a) / n
    s = 0
    for i in range(n):
        s += f(a + i * h)
    return s * h


def right_rectangle(f, a, b, n):
    h = (b - a) / n
    s = 0
    for i in range(1, n + 1):
        s += f(a + i * h)
    return s * h


def mid_rectangle(f, a, b, n):
    h = (b - a) / n
    s = 0
    for i in range(n):
        s += f(a + (i + 0.5) * h)
    return s * h


def trapezoidal(f, a, b, n):
    h = (b - a) / n
    s = (f(a) + f(b)) / 2
    for i in range(1, n):
        s += f(a + i * h)
    return s * h


def simpson(f, a, b, n):
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        if i % 2 == 0:
            s += 2 * f(a + i * h)
        else:
            s += 4 * f(a + i * h)
    return s * h / 3


def integrate_with_runge(method, f, a, b, tol, order):
    n = 4
    I_n = method(f, a, b, n)
    while True:
        n *= 2
        I_2n = method(f, a, b, n)
        error_estimate = abs(I_2n - I_n) / (2 ** order - 1)
        if error_estimate < tol:
            return I_2n, n
        I_n = I_2n

functions = {
        "1": ("f(x) = x^3", lambda x: x ** 3),
        "2": ("f(x) = 2*x^3 - 9*x^2 - 7*x + 11", lambda x: 2*x**3 - 9*x**2 - 7*x + 11),
        "3": ("f(x) = exp(x)", lambda x: math.exp(x)),
        "4": ("f(x) = 1/sqrt(x-1)", lambda x: 1/math.sqrt(x-1)),
        "5": ("f(x) = 1/(3-x)", lambda x: 1/(3-x)),
        "6": ("f(x) = 1/sqrt(x)", lambda x: math.log(x)/math.sqrt(x-1)**3)
    }

methods = {
    "1": ("Левые прямоугольники", left_rectangle, 1 ),
    "2": ("Правые прямоугольники", right_rectangle, 1),
    "3": ("Средние прямоугольники", mid_rectangle, 2),
    "4": ("Трапеции", trapezoidal, 2),
    "5": ("Симпсона", simpson, 4)
}


def parse_float(user_input):
    try:
        return float(user_input.replace(',', '.'))
    except ValueError:
        return None


def find_discontinuities(f, a, b, samples=1000, threshold=1e8):
    disc = []
    dx = (b - a) / samples
    for i in range(samples + 1):
        x = a + i * dx
        try:
            y = f(x)
            if math.isnan(y) or math.isinf(y) or abs(y) > threshold:
                disc.append(x)
        except Exception:
            disc.append(x)
    disc.sort()
    filtered = []
    if disc:
        filtered.append(disc[0])
        for x in disc[1:]:
            if abs(x - filtered[-1]) > dx:
                filtered.append(x)
    return filtered

def try_to_compute(f, x, threshold=1e4):
    try:
        y = f(x)
        if math.isnan(y) or math.isinf(y) or abs(y) > threshold:
            return None
        return y
    except Exception:
        return None

def main():
    print("Для выхода введите 'exit' при любом запросе ввода.")
    while True:
        print("Доступные функции для интегрирования:")
        for key, (desc, _) in functions.items():
            print(f"  {key}: {desc}")
        func_choice = input("Введите номер функции для интегрирования: ").strip()
        if func_choice.lower() == "exit":
            break
        if func_choice not in functions:
            print("Неверный выбор функции. Попробуйте снова.")
            continue
        f_desc, f = functions[func_choice]

        a_input = input("Введите нижний предел интегрирования a: ").strip()
        if a_input.lower() == "exit":
            break
        a = parse_float(a_input)
        if a is None:
            print("Некорректное значение нижнего предела. Попробуйте снова.")
            continue

        b_input = input("Введите верхний предел интегрирования b: ").strip()
        if b_input.lower() == "exit":
            break
        b = parse_float(b_input)
        if b is None:
            print("Некорректное значение верхнего предела. Попробуйте снова.")
            continue

        tol_input = input("Введите требуемую точность: ").strip()
        if tol_input.lower() == "exit":
            break
        tol = parse_float(tol_input)
        if tol is None or tol <= 0:
            print("Некорректное значение точности. Оно должно быть положительным числом.")
            continue
        disc = find_discontinuities(f, a, b)
        if not disc:
            print("Точек разрыва не обнаружено.")
            print("Выберите метод интегрирования:")
            for key, (desc, _, _) in methods.items():
                print(f"  {key}: {desc}")
            method_choice = input("Ваш выбор метода: ").strip()
            if method_choice.lower() == "exit":
                break
            if method_choice not in methods:
                print("Неверный выбор метода. Попробуйте снова.")
                continue
            method_desc, method_func, order = methods[method_choice]

            result, n_final = integrate_with_runge(method_func, f, a, b, tol, order)

            print("Результаты:")
            print(f"Функция: {f_desc}")
            print(f"Метод: {method_desc}")
            print(f"Интеграл = {result}")
            print(f"Число разбиений для достижения точности: {n_final}")
        else:
            print("Найдены предполагаемые точки разрыва")
            cov = False
            eps = 0.0001
            for d in disc:
                left, right = try_to_compute(f, d-eps), try_to_compute(f, d+eps)
                if a == d:
                    left = 0
                if b == d:
                    right = 0
                if left is None or right is None:
                    cov = True
                    break
            if cov:
                print("Интеграл расходится")
            else:
                if not a in disc:
                    disc.append(a-eps)
                if not b in disc:
                    disc.append(b+eps)
                disc.sort()
                for method in methods.values():
                    method_desc, method_func, order = method
                    print(f"Метод {method_desc}")
                    result = 0
                    for i in range(len(disc)-1):
                        su, _ = integrate_with_runge(method_func, f, disc[i]+eps, disc[i+1]-eps, tol, order)
                        result += su
                    print(f"Вычисленное значение интегралла: {result}")
if __name__ == '__main__':
    main()
