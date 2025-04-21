import tkinter as tk
from tkinter import filedialog, messagebox
import csv
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def solve_linear_system(A, b):
    n = len(b)
    M = [row[:] + [b_i] for row, b_i in zip(A, b)]
    for k in range(n):
        i_max = max(range(k, n), key=lambda i: abs(M[i][k]))
        M[k], M[i_max] = M[i_max], M[k]
        if abs(M[k][k]) < 1e-12:
            raise ValueError("Матрица вырождена")
        pivot = M[k][k]
        for j in range(k, n+1):
            M[k][j] /= pivot
        for i in range(n):
            if i != k:
                factor = M[i][k]
                for j in range(k, n+1):
                    M[i][j] -= factor * M[k][j]
    return [M[i][-1] for i in range(n)]

#линейное приближение коээфициентики
def linear_regression(xs, ys):
    n = len(xs)
    xm = sum(xs)/n
    ym = sum(ys)/n
    Sxx = sum((x-xm)**2 for x in xs)
    Sxy = sum((x-xm)*(y-ym) for x,y in zip(xs,ys))
    b = Sxy / Sxx
    a = ym - b * xm
    Syy = sum((y-ym)**2 for y in ys)
    r = Sxy / math.sqrt(Sxx * Syy) if Sxx>0 and Syy>0 else 0.0
    return a, b, r

#коэффициентики
def poly_regression(xs, ys, degree):
    n = len(xs)
    A = [[sum((x**(j+k)) for x in xs) for k in range(degree+1)]
         for j in range(degree+1)]
    b = [sum(y * (x**j) for x,y in zip(xs,ys)) for j in range(degree+1)]
    coeffs = solve_linear_system(A, b)
    return coeffs

def compute_metrics(xs, ys, func):
    n = len(xs)
    y_preds = [func(x) for x in xs]
    S = sum((y - yp)**2 for y,yp in zip(ys,y_preds))
    sigma = math.sqrt(S / n)
    ym = sum(ys)/n
    St = sum((y-ym)**2 for y in ys)
    R2 = 1 - S/St if St>0 else 0.0
    return S, sigma, R2, y_preds

def make_models(xs, ys):
    models = {}

    #линейная
    a, b, r = linear_regression(xs, ys)
    f_lin = lambda x: a + b*x
    S, sigma, R2, yp = compute_metrics(xs, ys, f_lin)
    models['Линейная'] = {
        'func': f_lin, 'coeffs': [a, b],
        'S': S, 'sigma': sigma, 'R2': R2, 'r': r, 'y_pred': yp,
        'expr': f'f(x) = {a:.4f} + {b:.4f}·x'
    }

    #полином 2-й
    c2 = poly_regression(xs, ys, 2)
    f_p2 = lambda x: c2[0] + c2[1]*x + c2[2]*x**2
    S, sigma, R2, yp = compute_metrics(xs, ys, f_p2)
    models['Полином 2'] = {
        'func': f_p2, 'coeffs': c2,
        'S': S, 'sigma': sigma, 'R2': R2, 'y_pred': yp,
        'expr': f'f(x) = {c2[0]:.4f} + {c2[1]:.4f}·x + {c2[2]:.4f}·x²'
    }

    #полином 3-й
    c3 = poly_regression(xs, ys, 3)
    f_p3 = lambda x: c3[0] + c3[1]*x + c3[2]*x**2 + c3[3]*x**3
    S, sigma, R2, yp = compute_metrics(xs, ys, f_p3)
    models['Полином 3'] = {
        'func': f_p3, 'coeffs': c3,
        'S': S, 'sigma': sigma, 'R2': R2, 'y_pred': yp,
        'expr': f'f(x) = {c3[0]:.4f} + {c3[1]:.4f}·x + {c3[2]:.4f}·x² + {c3[3]:.4f}·x³'
    }

    #экспоненциальная y = A·e^(B x)
    # ln y = ln A + B x
    if all(y>0 for y in ys):
        ls, bs, _ = linear_regression(xs, [math.log(y) for y in ys])
        A = math.exp(ls); B = bs
        f_exp = lambda x: A * math.exp(B*x)
        S, sigma, R2, yp = compute_metrics(xs, ys, f_exp)
        models['Экспоненциальная'] = {
            'func': f_exp, 'coeffs': [A, B],
            'S': S, 'sigma': sigma, 'R2': R2, 'y_pred': yp,
            'expr': f'f(x) = {A:.4f}·e^({B:.4f}·x)'
        }

    #логарифмическая y = A + B·ln(x)
    if all(x>0 for x in xs):
        ls, bs, _ = linear_regression([math.log(x) for x in xs], ys)
        A, B = ls, bs
        f_log = lambda x: A + B*math.log(x)
        S, sigma, R2, yp = compute_metrics(xs, ys, f_log)
        models['Логарифмическая'] = {
            'func': f_log, 'coeffs': [A, B],
            'S': S, 'sigma': sigma, 'R2': R2, 'y_pred': yp,
            'expr': f'f(x) = {A:.4f} + {B:.4f}·ln(x)'
        }

    #степенная y = A·x^B
    if all(x>0 and y>0 for x,y in zip(xs,ys)):
        ls, bs, _ = linear_regression(
            [math.log(x) for x in xs],
            [math.log(y) for y in ys]
        )
        A = math.exp(ls); B = bs
        f_pow = lambda x: A * x**B
        S, sigma, R2, yp = compute_metrics(xs, ys, f_pow)
        models['Степенная'] = {
            'func': f_pow, 'coeffs': [A, B],
            'S': S, 'sigma': sigma, 'R2': R2, 'y_pred': yp,
            'expr': f'f(x) = {A:.4f}·x^{B:.4f}'
        }

    return models

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("МНК‑Аппроксимация")
        self.geometry("1000x700")

        left = tk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        tk.Label(left, text="Ввести точки (x,y) по одной на строке:").pack()
        self.txt = tk.Text(left, width=50, height=20)
        self.txt.pack()

        btn_frame = tk.Frame(left)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Загрузить CSV", command=self.load_csv).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Compute", command=self.on_compute).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Сохранить отчёт", command=self.save_report).pack(side=tk.LEFT)

        tk.Label(left, text="Отчёт:").pack(pady=(10,0))
        self.out = tk.Text(left, width=50, height=25)
        self.out.pack()

        self.fig = plt.Figure(figsize=(6, 5), facecolor="#d0d8e6")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#d0d8e6")

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=self)
        widget = self.mpl_canvas.get_tk_widget()
        widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.last_models = None
        self.last_data = None
        self.colors = ['red','green','blue','orange','purple','brown']

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not path: return
        try:
            pts = []
            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row)>=2:
                        pts.append((float(row[0]), float(row[1])))
            self.txt.delete("1.0", tk.END)
            for x,y in pts:
                self.txt.insert(tk.END, f"{x},{y}\n")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить CSV:\n{e}")

    def on_compute(self):
        pts = []
        for line in self.txt.get("1.0", tk.END).splitlines():
            line = line.strip()
            if not line: continue
            try:
                x_str,y_str = line.split(",")
                pts.append((float(x_str), float(y_str)))
            except:
                messagebox.showerror("Ошибка", f"Неверный формат: '{line}'")
                return
        if not(8 <= len(pts) <= 12):
            messagebox.showerror("Ошибка", "Нужно от 8 до 12 точек")
            return

        xs, ys = zip(*pts)
        models = make_models(xs, ys)
        self.last_models = models
        self.last_data = (xs, ys)

        best = min(models.items(), key=lambda kv: kv[1]['sigma'])
        best_name = best[0]

        self.out.delete("1.0", tk.END)
        for name, m in models.items():
            self.out.insert(tk.END, f"{name}:\n")
            self.out.insert(tk.END, f"  {m['expr']}\n")
            self.out.insert(tk.END, f"  Коэфф.: {[round(c,6) for c in m['coeffs']]}\n")
            self.out.insert(tk.END, f"  σ = {m['sigma']:.6f}\n")
            self.out.insert(tk.END, f"  R² = {m['R2']:.6f}\n")
            if name=="Линейная":
                self.out.insert(tk.END, f"  r = {m['r']:.6f}\n")
            self.out.insert(tk.END, f"  S = {m['S']:.6f}\n\n")
        self.out.insert(tk.END, f"Лучшая модель: {best_name}\n")

        self.draw_plot()

    def draw_plot(self):
        xs, ys = self.last_data
        models = self.last_models

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        delta_y = ymax+ymin
        dx = (xmax - xmin) * 0.1
        xmin, xmax = xmin - dx, xmax + dx
        sample_x = [xmin + i * (xmax - xmin) / 200 for i in range(201)]

        self.ax.clear()
        self.ax.set_facecolor("#d0d8e6")

        self.ax.plot(xs, ys, 'o', label="Вводные точки", markersize=6)

        for name, m in models.items():
            sample_y = [m['func'](x) for x in sample_x]
            allowed_x = []
            allowed_y = []
            for i in range(len(sample_y)):
                if True:
                    allowed_y.append(sample_y[i])
                    allowed_x.append(sample_x[i])

            self.ax.plot(allowed_x, allowed_y, label=name, linewidth=2)

        self.ax.set_title("Приближение функции различными методами", pad=15)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True, linestyle="--", alpha=0.5)
        leg = self.ax.legend(frameon=True, facecolor="white", edgecolor="black", fontsize=9)
        for text in leg.get_texts():
            text.set_family("sans-serif")

        self.mpl_canvas.draw()

    def save_report(self):
        if not self.last_models:
            messagebox.showinfo("Внимание", "Сначала нажмите Compute")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                            filetypes=[("Text files","*.txt")])
        if not path: return
        try:
            with open(path, "w", encoding='utf-8') as f:
                f.write(self.out.get("1.0", tk.END))
            messagebox.showinfo("Готово", "Отчёт сохранён")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
