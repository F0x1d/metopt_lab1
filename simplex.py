# -*- coding: utf-8 -*-
"""
Решение задачи линейного программирования симплекс-методом
Автор: Зотеев Максим Евгеньевич
Поток: 1.2
"""


class SimplexSolver:
    """Класс для решения задачи линейного программирования симплекс-методом"""
    
    # Константы для численной стабильности
    EPSILON = 1e-9  # Порог для определения нуля
    MAX_ITERATIONS = 1000  # Максимальное количество итераций
    
    def __init__(self):
        self.objective_type = None  # 'min' или 'max'
        self.c = []  # Коэффициенты целевой функции (преобразованные для симплекс-метода)
        self.c_original = []  # Исходные коэффициенты (до преобразования max->min)
        self.num_vars = 0  # Количество исходных переменных
        self.num_constraints = 0  # Количество ограничений
        self.constraints = []  # Список ограничений [(коэффициенты, тип, правая_часть)]
        
        # Симплекс-таблица
        self.tableau = []  # Симплекс-таблица
        self.basis = []  # Индексы базисных переменных
        self.num_total_vars = 0  # Общее количество переменных (включая дополнительные)
        
    def read_from_file(self, filename):
        """Чтение задачи ЗЛП из файла"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Первая строка: тип задачи (min или max)
            self.objective_type = lines[0].strip()
            
            # Вторая строка: коэффициенты целевой функции
            self.c_original = list(map(float, lines[1].strip().split()))
            self.c = self.c_original.copy()
            
            # Если максимизация, преобразуем в минимизацию (max Z = min -Z)
            if self.objective_type == 'max':
                self.c = [-c for c in self.c_original]
                print("Преобразование max в min: новые коэффициенты =", self.c)
            
            self.num_vars = len(self.c)
            
            # Третья строка: количество переменных (для проверки)
            num_vars_check = int(lines[2].strip())
            
            # Четвертая строка: количество ограничений
            self.num_constraints = int(lines[3].strip())
            
            # Остальные строки: ограничения
            for i in range(4, 4 + self.num_constraints):
                parts = lines[i].strip().split()
                coeffs = list(map(float, parts[:-2]))
                constraint_type = parts[-2]
                rhs = float(parts[-1])
                self.constraints.append((coeffs, constraint_type, rhs))
            
            print("Задача успешно загружена из файла")
            print(f"Тип: {self.objective_type}")
            print(f"Целевая функция: Z = {' + '.join([f'{c}*x{i+1}' for i, c in enumerate(self.c_original)])}")
            print(f"Количество ограничений: {self.num_constraints}\n")
            
            for i, (coeffs, ctype, rhs) in enumerate(self.constraints):
                constraint_str = " + ".join([f"{coeffs[j]}*x{j+1}" for j in range(len(coeffs))])
                print(f"Ограничение {i+1}: {constraint_str} {ctype} {rhs}")
            
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")
            raise
    
    def build_initial_tableau(self):
        """Построение начальной симплекс-таблицы"""
        print("\n=== Построение начальной симплекс-таблицы ===")
        
        # Определяем количество дополнительных переменных
        num_slack = 0
        num_surplus = 0
        num_artificial = 0
        
        for coeffs, constraint_type, rhs in self.constraints:
            if constraint_type == '<=':
                num_slack += 1
            elif constraint_type == '>=':
                num_surplus += 1
                num_artificial += 1
            elif constraint_type == '=':
                num_artificial += 1
        
        self.num_total_vars = self.num_vars + num_slack + num_surplus + num_artificial
        
        print(f"Исходных переменных: {self.num_vars}")
        print(f"Остаточных переменных: {num_slack}")
        print(f"Излишних переменных: {num_surplus}")
        print(f"Искусственных переменных: {num_artificial}")
        print(f"Всего переменных: {self.num_total_vars}")
        
        # Создаем начальную симплекс-таблицу
        # Формат: [a11, a12, ..., a1n, b1]
        # Последняя строка - строка целевой функции
        
        self.tableau = []
        self.basis = []
        
        slack_idx = self.num_vars
        surplus_idx = self.num_vars + num_slack
        artificial_idx = self.num_vars + num_slack + num_surplus
        artificial_vars = []
        
        # Построение строк ограничений
        for i, (coeffs, constraint_type, rhs) in enumerate(self.constraints):
            row = coeffs.copy()
            
            # Обрабатываем правую часть (если отрицательная)
            if rhs < 0:
                rhs = -rhs
                row = [-c for c in row]
                # Меняем тип ограничения
                if constraint_type == '<=':
                    constraint_type = '>='
                elif constraint_type == '>=':
                    constraint_type = '<='
            
            # Добавляем остаточные переменные для всех ограничений
            for j in range(num_slack):
                row.append(0.0)
            
            # Добавляем излишние переменные для всех ограничений
            for j in range(num_surplus):
                row.append(0.0)
            
            # Добавляем искусственные переменные для всех ограничений
            for j in range(num_artificial):
                row.append(0.0)
            
            # Теперь устанавливаем нужные коэффициенты
            current_slack = slack_idx
            current_surplus = surplus_idx
            current_artificial = artificial_idx
            
            for j, (c, ct, r) in enumerate(self.constraints):
                if j < i:
                    if ct == '<=':
                        current_slack += 1
                    elif ct == '>=':
                        current_surplus += 1
                        current_artificial += 1
                    elif ct == '=':
                        current_artificial += 1
            
            if constraint_type == '<=':
                row[current_slack] = 1.0
                self.basis.append(current_slack)
            elif constraint_type == '>=':
                row[current_surplus] = -1.0
                row[current_artificial] = 1.0
                self.basis.append(current_artificial)
                artificial_vars.append(current_artificial)
            elif constraint_type == '=':
                row[current_artificial] = 1.0
                self.basis.append(current_artificial)
                artificial_vars.append(current_artificial)
            
            row.append(rhs)  # Правая часть
            self.tableau.append(row)
        
        # Строка целевой функции для фазы 1 (если есть искусственные переменные)
        if artificial_vars:
            print(f"\nИспользуем двухфазный симплекс-метод")
            print(f"Искусственные переменные: {[f'x{v+1}' for v in artificial_vars]}")
            
            # Фаза 1: минимизируем сумму искусственных переменных
            z_row = [0.0] * (self.num_total_vars + 1)
            for art_var in artificial_vars:
                z_row[art_var] = 1.0
            
            # Пересчитываем строку целевой функции (исключаем базисные переменные)
            for i, basis_var in enumerate(self.basis):
                if basis_var in artificial_vars:
                    for j in range(self.num_total_vars + 1):
                        z_row[j] -= self.tableau[i][j]
            
            self.tableau.append(z_row)
            
            # Решаем фазу 1
            print("\n=== ФАЗА 1: Поиск начального допустимого решения ===")
            if not self._simplex_iterations(phase=1):
                return False
            
            # Проверяем значение целевой функции фазы 1
            if abs(self.tableau[-1][-1]) > self.EPSILON:
                print(f"\nЗначение целевой функции фазы 1: {self.tableau[-1][-1]:.6f}")
                print("ЗАДАЧА НЕ ИМЕЕТ ДОПУСТИМЫХ РЕШЕНИЙ!")
                print("Ограничения противоречивы - не существует точки, удовлетворяющей всем ограничениям одновременно")
                return False
            
            print(f"\nФаза 1 завершена. Найдено допустимое базисное решение.")
            
            # Переходим к фазе 2
            print("\n=== ФАЗА 2: Решение основной задачи ===")
            
            # Заменяем строку целевой функции на исходную
            # Для симплекс-метода всегда работаем с -Z (минимизация)
            # Для максимизации инвертируем знаки коэффициентов (max Z = min -Z)
            z_row = [-c for c in self.c]  # Всегда используем -c для симплекс-метода
            
            # Дополняем нулями до нужной длины
            while len(z_row) < self.num_total_vars:
                z_row.append(0.0)
            z_row.append(0.0)  # Для правой части
            
            # Пересчитываем строку целевой функции (исключаем базисные переменные)
            for i, basis_var in enumerate(self.basis):
                if basis_var < self.num_vars:  # Только для исходных переменных
                    coeff = z_row[basis_var]  # Исправлено: убран лишний минус
                    for j in range(self.num_total_vars + 1):
                        z_row[j] -= coeff * self.tableau[i][j]
            
            self.tableau[-1] = z_row
            
        else:
            # Если искусственных переменных нет, сразу создаем строку целевой функции
            print(f"\nИспользуем обычный симплекс-метод (искусственные переменные не требуются)")
            
            # Для максимизации инвертируем коэффициенты (превращаем max в min)
            z_row = []
            if self.objective_type == 'min':
                z_row = [-c for c in self.c]
            else:
                z_row = [-c for c in self.c]  # Инвертируем для максимизации
            
            while len(z_row) < self.num_total_vars:
                z_row.append(0.0)
            z_row.append(0.0)
            
            self.tableau.append(z_row)
        
        print(f"\nНачальный базис: {[f'x{v+1}' for v in self.basis]}")
        return True
    
    def _simplex_iterations(self, phase=2):
        """Выполнение симплекс-итераций"""
        iteration = 0
        max_iterations = self.MAX_ITERATIONS
        
        while iteration < max_iterations:
            iteration += 1
            
            # Находим входящую переменную
            # Фаза 1: ищем наиболее отрицательный коэффициент (минимизация искусственных переменных)
            # Фаза 2 (для min): ищем наиболее положительный коэффициент (работаем с -Z)
            z_row = self.tableau[-1]
            entering_col = -1
            
            # В фазе 2 рассматриваем только исходные переменные
            search_range = self.num_total_vars if phase == 1 else self.num_vars
            
            if phase == 1:
                # Фаза 1: ищем наиболее отрицательный коэффициент
                min_coeff = -self.EPSILON
                for j in range(search_range):
                    if z_row[j] < min_coeff:
                        min_coeff = z_row[j]
                        entering_col = j
                
                # Если все коэффициенты неотрицательны, оптимум достигнут
                if entering_col == -1:
                    print(f"Итерация {iteration}: Оптимальное решение найдено")
                    return True
            else:
                # Фаза 2: для минимизации ищем наиболее положительный коэффициент
                # (так как работаем с -Z, положительный коэффициент означает улучшение)
                max_coeff = self.EPSILON
                for j in range(search_range):
                    if z_row[j] > max_coeff:
                        max_coeff = z_row[j]
                        entering_col = j
                
                # Если все коэффициенты неположительны, оптимум достигнут
                if entering_col == -1:
                    print(f"Итерация {iteration}: Оптимальное решение найдено")
                    return True
            
            print(f"\nИтерация {iteration}: Входящая переменная x{entering_col + 1}")
            
            # Находим выходящую переменную (минимальное отношение)
            leaving_row = -1
            min_ratio = float('inf')
            
            for i in range(self.num_constraints):
                if self.tableau[i][entering_col] > self.EPSILON:
                    ratio = self.tableau[i][-1] / self.tableau[i][entering_col]
                    if ratio < min_ratio:
                        min_ratio = ratio
                        leaving_row = i
            
            if leaving_row == -1:
                if phase == 1:
                    print("ОШИБКА: Не удалось найти допустимое решение в фазе 1")
                    print("Это может указывать на проблему в формулировке задачи")
                else:
                    print("Задача неограничена!")
                    print(f"Переменная x{entering_col + 1} может увеличиваться бесконечно")
                    print("Целевая функция может принимать сколь угодно малые значения")
                return False
            
            leaving_var = self.basis[leaving_row]
            print(f"Выходящая переменная: x{leaving_var + 1} (строка {leaving_row + 1})")
            
            # Обновляем базис
            self.basis[leaving_row] = entering_col
            
            # Выполняем поворот (преобразование Гаусса)
            pivot = self.tableau[leaving_row][entering_col]
            
            # Делим опорную строку на опорный элемент
            for j in range(self.num_total_vars + 1):
                self.tableau[leaving_row][j] /= pivot
            
            # Обнуляем столбец входящей переменной во всех остальных строках
            for i in range(len(self.tableau)):
                if i != leaving_row:
                    factor = self.tableau[i][entering_col]
                    for j in range(self.num_total_vars + 1):
                        self.tableau[i][j] -= factor * self.tableau[leaving_row][j]
            
            # Выводим текущее значение целевой функции
            current_z = -self.tableau[-1][-1]
            print(f"Текущее значение Z: {current_z:.6f}")
        
        print(f"\nДостигнуто максимальное количество итераций ({max_iterations})")
        return False
    
    def solve(self):
        """Решение задачи ЗЛП"""
        print("\n" + "="*70)
        print("РЕШЕНИЕ ЗАДАЧИ ЛИНЕЙНОГО ПРОГРАММИРОВАНИЯ")
        print("="*70)
        
        # Построение начальной симплекс-таблицы
        if not self.build_initial_tableau():
            return None
        
        # Выполнение симплекс-итераций (фаза 2)
        if not self._simplex_iterations(phase=2):
            return None
        
        # Извлечение решения
        solution = [0.0] * self.num_vars
        for i, basis_var in enumerate(self.basis):
            if basis_var < self.num_vars:
                solution[basis_var] = self.tableau[i][-1]
        
        # Вычисление значения целевой функции
        # Учитываем, что для max задач мы инвертировали коэффициенты
        z_value = sum(self.c[i] * solution[i] for i in range(self.num_vars))
        if self.objective_type == 'max':
            z_value = -z_value  # Инвертируем обратно для max задач
        
        return solution, z_value
    
    def save_result(self, result, filename='output.txt'):
        """Сохранение результата в файл"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("РЕЗУЛЬТАТ РЕШЕНИЯ ЗАДАЧИ ЛИНЕЙНОГО ПРОГРАММИРОВАНИЯ\n")
                f.write("="*70 + "\n\n")
                
                f.write("Исходная задача:\n")
                f.write(f"{self.objective_type} Z = ")
                f.write(" + ".join([f"{self.c_original[i]}*x{i+1}" for i in range(self.num_vars)]))
                f.write("\n\nПри ограничениях:\n")
                for i, (coeffs, ctype, rhs) in enumerate(self.constraints):
                    constraint_str = " + ".join([f"{coeffs[j]}*x{j+1}" for j in range(len(coeffs))])
                    f.write(f"{constraint_str} {ctype} {rhs}\n")
                f.write(f"xi >= 0, i = 1..{self.num_vars}\n")
                
                f.write("\n" + "-"*70 + "\n\n")
                
                if result is None:
                    f.write("РЕШЕНИЕ:\n")
                    f.write("Задача не имеет допустимых решений или неограничена.\n")
                else:
                    solution, z_value = result
                    f.write("РЕШЕНИЕ:\n\n")
                    f.write("Оптимальная точка:\n")
                    for i, val in enumerate(solution):
                        f.write(f"x{i+1} = {val:.6f}\n")
                    f.write(f"\nЗначение целевой функции:\n")
                    f.write(f"Z = {z_value:.6f}\n")
                
                f.write("\n" + "="*70 + "\n")
            
            print(f"\nРезультат сохранен в файл {filename}")
        except Exception as e:
            print(f"Ошибка при сохранении результата: {e}")
    
    def print_result(self, result):
        """Вывод результата на экран"""
        print("\n" + "="*70)
        print("ИТОГОВОЕ РЕШЕНИЕ")
        print("="*70)
        
        if result is None:
            print("\nЗадача не имеет допустимых решений или неограничена")
        else:
            solution, z_value = result
            print("\nОптимальная точка:")
            for i, val in enumerate(solution):
                print(f"  x{i+1} = {val:.6f}")
            print(f"\nЗначение целевой функции:")
            print(f"  Z = {z_value:.6f}")
        
        print("="*70)


def main():
    """Основная функция программы"""
    print("Программа решения задач линейного программирования")
    print("Автор: Зотеев Максим Евгеньевич, Поток 1.2")
    print()
    
    # Создание объекта решателя
    solver = SimplexSolver()
    
    # Чтение из файла
    input_file = 'input.txt'
    solver.read_from_file(input_file)
    
    # Решение задачи
    result = solver.solve()
    
    # Вывод и сохранение результата
    solver.print_result(result)
    solver.save_result(result)


if __name__ == "__main__":
    main()
