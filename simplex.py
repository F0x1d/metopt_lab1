from fractions import Fraction


class SimplexSolver:
    """Класс для решения задачи линейного программирования симплекс-методом"""
    
    # Константы для численной стабильности
    EPSILON = 1e-9  # Порог для определения нуля
    MAX_ITERATIONS = 1000  # Максимальное количество итераций
    
    def __init__(self, use_fractions=False):
        self.objective_type = None  # 'min' или 'max'
        self.c = []  # Коэффициенты целевой функции (преобразованные для симплекс-метода)
        self.c_original = []  # Исходные коэффициенты (до преобразования max->min)
        self.num_vars = 0  # Количество исходных переменных
        self.num_constraints = 0  # Количество ограничений
        self.constraints = []  # Список ограничений [(коэффициенты, тип, правая_часть)]
        self.use_fractions = use_fractions  # Использовать ли рациональные числа для точности
        
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
            
            self.num_vars = len(self.c)
            
            # Третья строка: количество переменных (для проверки)
            num_vars_check = int(lines[2].strip())
            
            # Четвертая строка: количество ограничений
            self.num_constraints = int(lines[3].strip())
            
            # Остальные строки: ограничения
            for i in range(4, 4 + self.num_constraints):
                parts = lines[i].strip().split()
                if self.use_fractions:
                    coeffs = list(map(Fraction, parts[:-2]))
                    constraint_type = parts[-2]
                    rhs = Fraction(parts[-1])
                else:
                    coeffs = list(map(float, parts[:-2]))
                    constraint_type = parts[-2]
                    rhs = float(parts[-1])
                self.constraints.append((coeffs, constraint_type, rhs))
            
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")
            raise
    
    def build_initial_tableau(self):
        """Построение начальной симплекс-таблицы"""
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
            if not self._simplex_iterations(phase=1):
                return False
            
            # Проверяем значение целевой функции фазы 1
            if abs(self.tableau[-1][-1]) > self.EPSILON:
                print("Задача не имеет допустимых решений")
                return False
            
            # Переходим к фазе 2
            
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
                    return True
            
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
                    print("Не удалось найти допустимое решение")
                else:
                    print("Задача неограничена")
                return False
            
            leaving_var = self.basis[leaving_row]
            
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
        
        print(f"Достигнуто максимальное количество итераций ({max_iterations})")
        return False
    
    def solve(self):
        """Решение задачи ЗЛП"""
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
                if result is None:
                    f.write("Задача не имеет решения\n")
                else:
                    solution, z_value = result
                    for i, val in enumerate(solution):
                        f.write(f"{val:.6f}\n")
                    f.write(f"{z_value:.6f}\n")
        except Exception as e:
            print(f"Ошибка при сохранении результата: {e}")
    
    def print_result(self, result):
        """Вывод результата на экран"""
        if result is None:
            print("Задача не имеет решения")
        else:
            solution, z_value = result
            for i, val in enumerate(solution):
                print(f"{val:.6f}")
            print(f"{z_value:.6f}")


def main():
    """Основная функция программы"""
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
