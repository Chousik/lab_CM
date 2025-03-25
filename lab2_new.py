import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

matplotlib.use("TkAgg")
#функции
def f1(x):
    return -1.38 * x ** 3 - 5.42 * x ** 2 + 2.57 * x + 10.95
def f2(x):
    return x ** 3 - 1.89 * x ** 2 - 2 * x + 1.76
def f3(x):
    return x / 2 - 2 * ((x + 2.39) ** (1 / 3))
def f4(x):
    return -x / 2 + math.e ** x + 5 * math.sin(x)
def f5(x):
    return x ** 2 - 2
#производные первые
def df1(x):
    return -1.38 * 3 * x ** 2 - 5.42 * 2 * x + 2.57
def df2(x):
    return 3 * x ** 2 - 1.89 * 2 * x - 2
def df3(x):
    return 0.5 - 2 * (1.0 / 3.0) * ((x + 2.39) ** (-2.0 / 3.0))
def df4(x):
    return -0.5 + math.e ** x + 5 * math.cos(x)
def df5(x):
    return 2 * x
#производные вторые
def ddf1(x):
    return -1.38 * 3 * 2 * x - 5.42 * 2
def ddf2(x):
    return 3 * x * 2 - 1.89 * 2
def ddf3(x):
    return 2 * (1.0 / 3.0) * ((x + 2.39) ** (-2.0 / 3.0))
def ddf4(x):
    return  math.e ** x + -5 * math.sin(x)
def ddf5(x):
    return 2
functions_single = {
    "f1(x) = -1.38*x^3 - 5.42*x^2 + 2.57*x + 10.95": f1,
    "f2(x) = x^3 - 1.89*x^2 - 2*x + 1.76": f2,
    "f3(x) = x/2 - 2*(x+2.39)^(1/3)": f3,
    "f4(x) = -x/2 + e^x + 5*sin(x)": f4,
    "f5(x) = x^2 - 2": f5,
}
dr = {
    "f1(x)": df1,
    "f2(x)": df2,
    "f3(x)": df3,
    "f4(x)": df4,
    "f5(x)": df5,
}
ddr = {
    "f1(x)": ddf1,
    "f2(x)": ddf2,
    "f3(x)": ddf3,
    "f4(x)": ddf4,
    "f5(x)": ddf5,
}
#метод хорд
def chord_method(f, a, b, eps):
    log_iterations = []
    fa = f(a)
    fb = f(b)
    x_last = a
    if fa * fb > 0:
        raise ValueError("На отрезке [%g, %g] функция не меняет знак. Вероятно, нет корня или их несколько." % (a, b))
    iteration = 0
    while True:
        iteration += 1
        x = a-(a-b)/(fa-fb)*f(a)
        fx=f(x)
        log_str = f"{iteration}: a = {a:.3f}, b = {b:.3f}, x = {x:.3f}, f(x)={f(x):.6g}, |x_next - x_current| = {abs(x-x_last):.6g}"
        log_iterations.append(log_str)
        if abs(x-x_last) < eps:
            return x, f(x), iteration, log_iterations
        x_last = x
        if fa*fx<0:
            b=x
        else:
            a=x
        fa = f(a)
        fb = f(b)
#метод Ньютона
def newton_method(f, df, ddf, a, b, eps):
    log_iterations = []
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("На отрезке [%g, %g] нет гарантии единственного корня (функция не меняет знак)." % (a, b))
    if fa*ddf(a)>0:
        x_current = a
    else:
        x_current = b
    iteration = 0
    while True:
        iteration += 1
        fx = f(x_current)
        dfx = df(x_current)
        x_next = x_current - fx / dfx
        log_str = f"{iteration}: x = {x_next:.6f}, f(x) = {f(x_next):.6g}, |x_next - x_current| = {abs(x_next - x_current):.6g}"
        log_iterations.append(log_str)
        if abs(x_next - x_current) < eps:
            return x_next, f(x_next), iteration, log_iterations
        x_current = x_next
#метод_простых_иттераций
def simple_iteration_method_single(f, df, ddf, a, b, eps):
    log_iterations = []
    iteration = 0
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError(
            "На отрезке [%g, %g] функция не меняет знак. Возможна ошибка: нет корня или их несколько." % (a, b))
    if abs(df(a)) > abs(df(b)):
        alpha = 1/df(a)
    else:
        alpha = 1/df(b)
    def g(x):
        return x - alpha * f(x)
    def dg(x):
        return 1 - alpha * df(x)
    if max(dg(a), dg(b))>1:
            raise ValueError("Достаточное условие сходимости метода простой итерации может быть нарушено!")
    if fa*ddf(a)>0:
        x_current = a
    else:
        x_current = b
    while True:
        iteration += 1
        x_next = g(x_current)
        log_str = f"{iteration}: x = {x_next:.6f}, f(x)={f(x_next):.6g}, |x_next - x_current| = {abs(x_next - x_current):.6g}"
        log_iterations.append(log_str)
        if abs(x_next - x_current) < eps:
            return x_next, f(x_next), iteration, log_iterations
        x_current = x_next
def system1_f1_plot(x, y):
    return 0.1 * x ** 2 + x + 0.2 * y ** 2 - 0.3
def system1_f2_plot(x, y):
    return 0.2 * x ** 2 + y + 0.1 * x * y - 0.7
def system1_f1(x, y):
    return 0.3 - 0.1 * x ** 2 - 0.2 * y ** 2
def system1_f2(x, y):
    return 0.7 - 0.2 * x ** 2 - 0.1 * x * y
def system1_df1dx(x, y):
    return -0.2 * x
def system1_df1dy(x, y):
    return -0.4 * y
def system1_df2dx(x, y):
    return -0.4 * x - 0.1 * y
def system1_df2dy(x, y):
    return -0.1 * x

def system2_f1_plot(x, y):
    return x - (0.2 + 0.5*math.cos(y))
def system2_f2_plot(x, y):
    return y - (0.3 + 0.5*math.sin(x))
def system2_f1(x, y):
    return 0.2 + 0.5*math.cos(y)
def system2_f2(x, y):
    return 0.3 + 0.5*math.sin(x)
def system2_dg1dx(x, y):
    return 0
def system2_dg1dy(x, y):
    return -0.5*math.sin(y)
def system2_dg2dx(x, y):
    return 0.5*math.cos(x)
def system2_dg2dy(x, y):
    return 0

systems = {
    "Система 1:\n{0.1x^2 + x + 0.2y^2 - 0.3 = 0;\n 0.2x^2 + x + 0.1xy - 0.7 = 0}":
        (system1_f1, system1_f2, system1_f1_plot, system1_f2_plot, (system1_df1dx, system1_df1dy,
                                                                    system1_df2dx, system1_df2dy)),
    "Система 2:\n{x - 0.5cos(y) - 0.2;\n y - 0.5sin(x) - 0.3}":
        (system2_f1, system2_f2, system2_f1_plot, system2_f2_plot, (system2_dg1dx, system2_dg1dy,
                                                                    system2_dg2dx, system2_dg2dy)),
}
#метод иттерации систем
def simple_iteration_method_system(f1, f2, x0, y0, G, eps):
    f1dx, f1dy, f2dx, f2dy = G
    if (f1dx(x0, y0) + f1dy(x0, y0)) > 1 or (f2dx(x0, y0) + f2dy(x0, y0)) > 1:
        raise ValueError("Ошибка", f"Условия сходимости не выполняются, ошибка")
    log_iterations = []
    x_current = x0
    y_current = y0
    iteration = 0
    while True:
        iteration += 1
        x_next = f1(x_current, y_current)
        y_next = f2(x_current, y_current)
        dist = max(abs(y_next - y_current), abs(x_next - x_current))

        log_str = (f"{iteration}: x = {x_next:.6f}, y = {y_next:.6f}, "
                   f"|max(|x_next-x_current|,|y_next-y_current|)|={dist:.6g}")
        log_iterations.append(log_str)

        if dist < eps:
            return x_next, y_next, iteration, log_iterations
        x_current, y_current = x_next, y_next
class NonlinearSolverApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Решение нелинейных уравнений и систем")
        self.geometry("1200x1000")
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.single_frame = SingleEquationFrame(self.notebook)
        self.system_frame = SystemEquationFrame(self.notebook)
        self.notebook.add(self.single_frame, text="Нелинейные уравнения")
        self.notebook.add(self.system_frame, text="Системы уравнений")
class SingleEquationFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        label = ttk.Label(self, text="Решение одного нелинейного уравнения", font=("TkDefaultFont", 12, "bold"))
        label.pack(pady=5)
        self.func_label = ttk.Label(self, text="Выберите уравнение:")
        self.func_label.pack()
        self.func_var = tk.StringVar()
        self.func_combobox = ttk.Combobox(self, textvariable=self.func_var, values=list(functions_single.keys()),
                                          width=50)
        self.func_combobox.current(0)
        self.func_combobox.pack(pady=5)
        self.method_label = ttk.Label(self, text="Выберите метод:")
        self.method_label.pack()
        self.method_var = tk.StringVar()
        self.method_combobox = ttk.Combobox(self, textvariable=self.method_var,
                                            values=["Метод хорд", "Метод Ньютона", "Метод простой итерации"],
                                            width=30)
        self.method_combobox.current(0)
        self.method_combobox.pack(pady=5)
        frame_inputs = ttk.Frame(self)
        frame_inputs.pack(pady=5)
        self.label_a = ttk.Label(frame_inputs, text="Левая граница (a):")
        self.label_a.grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.entry_a = ttk.Entry(frame_inputs)
        self.entry_a.grid(row=0, column=1, padx=5, pady=2)
        self.label_b = ttk.Label(frame_inputs, text="Правая граница (b):")
        self.label_b.grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.entry_b = ttk.Entry(frame_inputs)
        self.entry_b.grid(row=1, column=1, padx=5, pady=2)
        self.label_eps = ttk.Label(frame_inputs, text="Точность (eps):")
        self.label_eps.grid(row=2, column=0, sticky="e", padx=5, pady=2)
        self.entry_eps = ttk.Entry(frame_inputs)
        self.entry_eps.insert(0, "0.01")
        self.entry_eps.grid(row=2, column=1, padx=5, pady=2)
        buttons_frame = ttk.Frame(self)
        buttons_frame.pack(pady=5)
        self.load_button = ttk.Button(buttons_frame, text="Загрузить из файла", command=self.load_from_file)
        self.load_button.grid(row=0, column=0, padx=5)
        self.save_button = ttk.Button(buttons_frame, text="Сохранить результат в файл", command=self.save_to_file)
        self.save_button.grid(row=0, column=1, padx=5)
        self.solve_button = ttk.Button(buttons_frame, text="Рассчитать", command=self.solve_equation)
        self.solve_button.grid(row=0, column=2, padx=5)
        self.output_text = tk.Text(self, height=15)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def load_from_file(self):
        filename = filedialog.askopenfilename(title="Выберите файл с исходными данными",
                                              filetypes=[("Текстовые файлы", "*.txt")])
        if not filename:
            return
        try:
            with open(filename, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) < 3:
                raise ValueError("В файле недостаточно строк (нужно минимум 3).")
            self.entry_a.delete(0, tk.END)
            self.entry_a.insert(0, lines[0].strip())
            self.entry_b.delete(0, tk.END)
            self.entry_b.insert(0, lines[1].strip())
            self.entry_eps.delete(0, tk.END)
            self.entry_eps.insert(0, lines[2].strip())
            messagebox.showinfo("Загрузка", "Данные успешно загружены из файла.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить данные: {e}")

    def save_to_file(self):
        result_text = self.output_text.get("1.0", tk.END).strip()
        if not result_text:
            messagebox.showinfo("Сохранение", "Нет результата для сохранения.")
            return
        filename = filedialog.asksaveasfilename(title="Сохранить результат",
                                                defaultextension=".txt",
                                                filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")])
        if not filename:
            return
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(result_text)
            messagebox.showinfo("Сохранение", "Результат успешно сохранён в файл.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")

    def solve_equation(self):
        self.output_text.delete("1.0", tk.END)
        try:
            a = float(self.entry_a.get().replace(",", "."))
            b = float(self.entry_b.get().replace(",", "."))
            eps = float(self.entry_eps.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректно введены числа a, b или eps.")
            return

        if a >= b:
            messagebox.showerror("Ошибка", "Левая граница должна быть меньше правой.")
            return
        func_key = self.func_var.get()
        method_key = self.method_var.get()
        try:
            try:
                f = functions_single[func_key]
            except Exception:
                raise ValueError("Неизвестная функция")
            if "f1(" in func_key:
                df = dr["f1(x)"]
                ddf = ddr["f1(x)"]
            elif "f2(" in func_key:
                df = dr["f2(x)"]
                ddf = ddr["f2(x)"]
            elif "f3(" in func_key:
                df = dr["f3(x)"]
                ddf = ddr["f3(x)"]
            elif "f4(" in func_key:
                df = dr["f4(x)"]
                ddf = ddr["f3(x)"]
            elif "f5(" in func_key:
                df = dr["f5(x)"]
                ddf = ddr["f3(x)"]
            if method_key == "Метод хорд":
                x_sol, f_sol, iters, logs = chord_method(f, a, b, eps)
            elif method_key == "Метод Ньютона":
                x_sol, f_sol, iters, logs = newton_method(f, df, ddf, a, b, eps)
            elif method_key == "Метод простой итерации":
                x_sol, f_sol, iters, logs = simple_iteration_method_single(f, df, ddf, a, b, eps)
            else:
                raise ValueError("Неизвестный метод.")
            self.output_text.insert(tk.END, "Процесс решения:\n")
            for l in logs:
                self.output_text.insert(tk.END, l + "\n")
            result_str = (f"\nРезультат:\nНайденный корень: {x_sol:.6f}\n"
                          f"Значение функции в корне: {f_sol:.6g}\n"
                          f"Число итераций: {iters}\n")
            self.output_text.insert(tk.END, result_str)
            self.plot_function(a, b, f, x_sol)
        except Exception as e:
            messagebox.showerror("Ошибка при вычислении", str(e))

    def plot_function(self, a, b, f, x_sol):
        self.figure.clear()
        plot = self.figure.add_subplot(111)
        n_points = 200
        x_vals = []
        y_vals = []
        step = (b - a) / n_points
        x_current = a
        for _ in range(n_points + 1):
            x_vals.append(x_current)
            y_vals.append(f(x_current))
            x_current += step
        plot.plot(x_vals, y_vals, label="f(x)")
        plot.axhline(0, color='black', linewidth=1)
        plot.scatter([x_sol], [f(x_sol)], color='red', label="Корень")
        plot.set_xlabel("x")
        plot.set_ylabel("f(x)")
        plot.set_title("График функции на интервале [{:.2f}, {:.2f}]".format(a, b))
        plot.legend()
        plot.grid(True)
        self.canvas.draw()

class SystemEquationFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        label = ttk.Label(self, text="Решение систем нелинейных уравнений", font=("TkDefaultFont", 12, "bold"))
        label.pack(pady=5)
        self.system_label = ttk.Label(self, text="Выберите систему уравнений:")
        self.system_label.pack()
        self.system_var = tk.StringVar()
        self.system_combobox = ttk.Combobox(self, textvariable=self.system_var, values=list(systems.keys()), width=50)
        self.system_combobox.current(0)
        self.system_combobox.pack(pady=5)
        frame_inputs = ttk.Frame(self)
        frame_inputs.pack(pady=5)
        self.label_x0 = ttk.Label(frame_inputs, text="Начальное приближение x0:")
        self.label_x0.grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.entry_x0 = ttk.Entry(frame_inputs)
        self.entry_x0.insert(0, "1.0")
        self.entry_x0.grid(row=0, column=1, padx=5, pady=2)
        self.label_y0 = ttk.Label(frame_inputs, text="Начальное приближение y0:")
        self.label_y0.grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.entry_y0 = ttk.Entry(frame_inputs)
        self.entry_y0.insert(0, "1.0")
        self.entry_y0.grid(row=1, column=1, padx=5, pady=2)
        self.label_eps = ttk.Label(frame_inputs, text="Точность (eps):")
        self.label_eps.grid(row=2, column=0, sticky="e", padx=5, pady=2)
        self.entry_eps = ttk.Entry(frame_inputs)
        self.entry_eps.insert(0, "0.01")
        self.entry_eps.grid(row=2, column=1, padx=5, pady=2)
        buttons_frame = ttk.Frame(self)
        buttons_frame.pack(pady=5)
        self.solve_button = ttk.Button(buttons_frame, text="Рассчитать (метод простой итерации)",
                                       command=self.solve_system)
        self.solve_button.grid(row=0, column=0, padx=5)
        self.output_text = tk.Text(self, height=15)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    def solve_system(self):
        self.output_text.delete("1.0", tk.END)
        system_key = self.system_var.get()
        f1, f2, f1_plot, f2_plot, G= systems[system_key]
        try:
            x0 = float(self.entry_x0.get().replace(",", "."))
            y0 = float(self.entry_y0.get().replace(",", "."))
            eps = float(self.entry_eps.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректно введены x0, y0 или eps.")
            return
        try:
            x_sol, y_sol, iters, logs = simple_iteration_method_system(f1, f2, x0, y0, G, eps)
            self.output_text.insert(tk.END, "Процесс решения:\n")
            for l in logs:
                self.output_text.insert(tk.END, l + "\n")
            f1_val = f1(x_sol, y_sol)
            f2_val = f2(x_sol, y_sol)
            result_str = (f"\nРезультат:\nНайденное решение (x, y) = ({x_sol:.6f}, {y_sol:.6f})\n"
                          f"Число итераций: {iters}\n"
                          f"f1(x,y) = {f1_val:.6g}, f2(x,y) = {f2_val:.6g}\n")
            self.output_text.insert(tk.END, result_str)
            self.output_text.insert(tk.END, f"Вектор погрешностей: ({abs(f1_val):.6g}, {abs(f2_val):.6g})\n")
            self.plot_system(f1_plot, f2_plot, x_sol, y_sol)
        except Exception as e:
            messagebox.showerror("Ошибка при вычислении", str(e))
    def plot_system(self, f1, f2, x_sol, y_sol):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        x_min, x_max = x_sol - 1.5, x_sol + 1.5
        y_min, y_max = y_sol - 1.5, y_sol + 1.5
        x_grid_count = 500
        y_grid_count = 500
        x_values = [x_min + i * (x_max - x_min) / x_grid_count for i in range(x_grid_count + 1)]
        y_values = [y_min + j * (y_max - y_min) / y_grid_count for j in range(y_grid_count + 1)]
        Z1 = []
        Z2 = []
        for y in y_values:
            row1 = []
            row2 = []
            for x in x_values:
                row1.append(f1(x, y))
                row2.append(f2(x, y))
            Z1.append(row1)
            Z2.append(row2)
        ax.contour(x_values, y_values, Z1, levels=[0], colors='blue')
        ax.contour(x_values, y_values, Z2, levels=[0], colors='green')
        ax.scatter([x_sol], [y_sol], color='red', label="Решение")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("График системы")
        ax.legend()
        ax.grid(True)
        self.canvas.draw()
def main():
    app = NonlinearSolverApp()
    app.mainloop()
if __name__ == "__main__":
    main()