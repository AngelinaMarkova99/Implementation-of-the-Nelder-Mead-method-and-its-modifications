import numpy as np
import matplotlib.pyplot as plt

from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import fabs
from numpy import sin

from inspect import getfullargspec
from random import randrange


def generate_n_vertices(func, num_points=0, multiplier=0.1,
                        interval=(-50, 50), dt=1):

    if num_points < 3:
        num_points = len(getfullargspec(func).args) + 1
    vertices = []
    start = interval[0]
    stop = interval[1]
    for i in range(0, num_points):
        point_x = randrange(start, stop, dt) * multiplier
        point_y = randrange(start, stop, dt) * multiplier
        point_a = randrange(start, stop, dt) * multiplier
        point_b = randrange(start, stop, dt) * multiplier
        point_c = randrange(start, stop, dt) * multiplier
        point_d = randrange(start, stop, dt) * multiplier

        vertices.append([point_x, point_y, point_a, point_b, point_c, point_d])
        # vertices.append([point_x, point_y, point_a, point_b]) - Для размерности 4
        # vertices.append([point_x, point_y]) - Для размерности 2

    return np.array(vertices)


def nelder_mead(func, start_points=None, max_iter=500, max_D=0.01,
                alpha=1, beta=0.5, gamma=2, num_points=0):
    # alpha - коэффициент отражения
    # beta - коэффициент сжатия
    # gamma - коэффициент растяжения

    # Автоматическое заполнение точек
    # Генератор случайных чисел
    if start_points is None:
        start_points = generate_n_vertices(func, num_points)

    # Стабильная единичная матрица

    if start_points is None:
        dim = len(getfullargspec(func).args)
        start_points = np.eye(dim)  # единичная матрица
        # start_points *= 3  # множитель для тестов
        # и + строка из нулей
        start_points = np.vstack((start_points, np.zeros((1, dim))))
    # print(f"Стартовые точки: \n{start_points}")

    num_points = start_points.shape[0]

    buff_points = start_points
    step_points = []  # для графика

    for i in range(0, max_iter):

        # Сортировка
        sorted_points = []
        for j in range(0, num_points):
            sorted_points.append([func(*buff_points[j]),
                                  *buff_points[j]])
        sorted_points.sort(key=lambda x: x[0], reverse=True)
        # print(f"Порядок точек по значению функции: \n{sorted_points}")
        buff_points = np.delete(sorted_points, 0, axis=1)
        step_points.append(buff_points)

        # Проверка сходимости по дисперсии точек
        if np.var(buff_points) < max_D:
            break

        # Определение точек для работы
        worst_point = buff_points[0]
        f_worst_point = sorted_points[0][0]
        worst_point2 = buff_points[1]
        f_worst_point2 = sorted_points[1][0]
        best_point = buff_points[-1]
        f_best_point = sorted_points[-1][0]
        # print(f"Худшая точка: {worst_point}")
        # print(f"Худшая точка №2: {worst_point2}")
        # print(f"Лучшая точка: {best_point}")

        # Центр тяжести (считается без худшей точки)
        gravity_point = (buff_points[1:]).sum(axis=0)
        gravity_point = gravity_point/(num_points-1)

        # Отражение
        reflected_point = (1+alpha)*gravity_point - alpha*worst_point
        f_reflected_point = func(*reflected_point)

        # Растяжение
        if f_reflected_point <= f_best_point:
            expanded_point = (1-gamma)*gravity_point + gamma*reflected_point
            f_expanded_point = func(*expanded_point)
            if f_expanded_point < f_reflected_point:
                buff_points[0] = expanded_point
            else:
                buff_points[0] = reflected_point
        # Присвоение без растяжения
        elif f_best_point < f_reflected_point <= f_worst_point2:
            buff_points[0] = reflected_point
        # Сжатие
        else:
            if f_worst_point2 < f_reflected_point <= f_worst_point:
                buff_points[0] = reflected_point
                worst_point = reflected_point
                f_worst_point = f_reflected_point
            contracted_point = (1-beta)*gravity_point + beta*worst_point
            f_contracted_point = func(*contracted_point)
            if f_contracted_point <= f_worst_point:
                buff_points[0] = contracted_point
            else:
                buff_points[:-1] = best_point \
                                           + (buff_points[:-1]
                                              - best_point) / 2
    # Конец цикла с проверками
    return buff_points[-1], func(*buff_points[-1]), step_points


def error_plots(func, f_min, num_points=tuple([3])):
    y_vals = np.logspace(-10, 0, 10)  # максимальная дисперсия

    fig = plt.figure('Plots')
    axes = fig.subplots(2, 1)
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Error')

    for curr_num_points in num_points:

        # # Средняя ошибка и среднее количество операций
        # x_vals_error_m = []
        # x_vals_iter_m = []

        # for i in y_vals:
        #     mean_error = []
        #     mean_iter = []
        #     for j in range(0, 50):  # выборка на каждую точность
        #         point, found_min, steps = nelder_mead(func, max_D=i, num_points=curr_num_points)
        #         mean_error.append(np.abs(f_min - found_min))
        #         mean_iter.append(len(steps))
        #     x_vals_error_m.append(np.mean(mean_error))
        #     x_vals_iter_m.append(np.mean(mean_iter))


        point, found_min, steps = nelder_mead(func, max_D=0.0001, num_points=curr_num_points)
        x_vals_error = []

        # print(steps)
        for j, fig in enumerate(steps, start=1):
            sorted_points = []
            closed_fig = np.vstack([fig, fig[0]])
            for var in range(0, len(closed_fig) - 1):
                sorted_points.append(func(closed_fig[var][0], closed_fig[var][1], closed_fig[var][2], closed_fig[var][3], closed_fig[var][4], closed_fig[var][5]))
            x_vals_error.append(abs((f_min - min(sorted_points))))

        print(x_vals_error)
        print("------")
        axes[0].semilogy(x_vals_error, label=f'{curr_num_points} points')

        axes[0].legend()
        axes[0].grid()
    plt.show()


def draw_nelder_mead(func, steps, x_interval=(-5, 5),
                     y_interval=(-5, 5), dt=0.1, pause=0.,
                     num_contour_lines=10):

    x_values = np.arange(x_interval[0], x_interval[1], dt)
    y_values = np.arange(y_interval[0], y_interval[1], dt)
    X, Y = np.meshgrid(x_values, y_values)
    
    fig = plt.figure('Steps')
    ax = fig.add_subplot(111)
    ax.contour(X, Y, func(X, Y), num_contour_lines)  # сама функция
    # Пошаговое рисование
    for i, fig in enumerate(steps, start=1):
        closed_fig = np.vstack([fig, fig[0]])
        ax.plot(closed_fig[:, 0], closed_fig[:, 1])
        ax.set_title(f"{i}-ый шаг")
        plt.draw()
        if pause > 0:
            plt.pause(pause)
    plt.show()


# Некоторые функции для проверки:
# def f(x, y): return x**2 + (2 * y**2) + 5 – простая функция, мин: 5 (0; 0)
# def f(x1, x2, x3): return x1**2 + x2**2 + x3**2 + 1 - простая функция, мин: 1 (0, 0, 0)
# def f(x1, x2): return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2 - Химмельблау
# мин: 0 (3, 2)(-2.80, 3.13)(-3.77, -3.28)(3.58, -1.84)
#
# def f(x1, x2): return (1 - x1)**2 + 100*(x2 - x1**2)**2 - Розенброк, мин: 0 (1, 1)
# def f(x1, x2, x3, x4): return (1 - x1)**2 + 100*(x2 - x1**2)**2 + (1 - x3)**2 + 100*(x4 - x3**2)**2 - Розенброк
# мин: 0 (1, 1, 1, 1)

# Функции, которые можно запустить:
# generate_n_vertices(функция, количество вершин, множитель для координат,
#                     интервал для координат, шаг координат в интервале)
# nelder_mead(функция, начальные точки (задать вручную), максимум итераций, максимум дисперсии,
#             коэффициент отражения, коэффициент сжатия, коэффициент растяжения,
#             количество точек для рандомной генерации (атоматически))
# error_plots(функция, минимум функции, количество точек в поиск(е/ах) минимума)
# draw_nelder_mead(функция, шаги, интервал х для графика, интервал у для графика,
#                  шаг в интервале, пауза отрисовки, количество контурных линий)

if __name__ == '__main__':
    # def f(x1, x2, x3, x4): return 100*(x1**2-x2)**2 + (x1-1)**2 + (x3-1)**2 + 90*(x3**2-x4)**2 + 10.1*((x2-1)**2 + (x4-1)**2) + 19.8*(x2-1)*(x4-1)

    def f(x1, x2, x3, x4, x5, x6): return 418.9829 * 6 - x1 * sin(sqrt(abs(x1))) - x2 * sin(sqrt(abs(x2)))- x3 * sin(sqrt(abs(x3)))- x4 * sin(sqrt(abs(x4)))- x5 * sin(sqrt(abs(x5)))- x6 * sin(sqrt(abs(x6)))

    # def f(x1, x2, x3, x4, x5, x6): return -60.0 * exp(-0.2 * sqrt(0.5 * (x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2 + x6 ** 2))) - exp(
    #     0.5 * (cos(2 * pi * x1) + cos(2 * pi * x2)+ cos(2 * pi * x3)+ cos(2 * pi * x4)+ cos(2 * pi * x5)+ cos(2 * pi * x6))) + e + 60

    # def f(x1, x2): return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2

    # def f(x1, x2, x3, x4, x5, x6): return (1 - x1)**2 + 100*(x2 - x1**2)**2 + (1 - x3)**2 + 100*(x4 - x3**2)**2 + (1 - x5)**2 + 100*(x6 - x5**2)**2

    point, f_min, steps = nelder_mead(f, max_D=0.01, num_points=4)
    print(f"Координаты точки: {point}")
    print(f"Значение минимума: {f_min}")
    print(f"Количество итераций: {len(steps)}")

    error_plots(f, 0, (7, 14, 49))

    # draw_nelder_mead(f, steps, pause=1, num_contour_lines=15)
    print("\033[32mDone\033[37m")
