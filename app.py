import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Инициализация состояния параметров
if 'params' not in st.session_state:
    st.session_state.params = {
        'lambda_': 10.0,
        'mu': 15.0,
        'gamma': 0.5,
        'delta': 2.0,
        'alpha': 1.0,
        'beta': 3.0
    }

# Функция для расчета вероятностей
def calculate_pi(lambda_, mu, gamma, delta, alpha, beta):
    A = np.array([
        [-lambda_, mu, delta, beta],
        [lambda_, -(mu + gamma + alpha), 0, 0],
        [0, gamma, -delta, 0],
        [0, alpha, 0, -beta],
        [1, 1, 1, 1]  # Условие нормировки
    ])
    
    b = np.array([0, 0, 0, 0, 1])
    
    try:
        # Пытаемся решить систему точно
        pi = np.linalg.solve(A[:4,:4], b[:4])
    except np.linalg.LinAlgError:
        # Если система вырождена, используем метод наименьших квадратов
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Нормируем вероятности
    pi /= np.sum(pi)
    return pi[:4]  # Возвращаем только π₀-π₃

# Интерфейс приложения
st.title("Модель установившегося режима СОИСН")
st.markdown("Анализ вероятностей состояний системы обработки информации специального назначения.")

# Боковая панель с параметрами
st.sidebar.header("Параметры системы")
params = st.session_state.params

params['lambda_'] = st.sidebar.slider(
    "λ (интенсивность входящего потока, 1/час):",
    0.1, 30.0, params['lambda_'], 0.1
)
params['mu'] = st.sidebar.slider(
    "μ (интенсивность обработки, 1/час):",
    0.1, 30.0, params['mu'], 0.1
)
params['gamma'] = st.sidebar.slider(
    "γ (интенсивность сбоев, 1/час):",
    0.01, 5.0, params['gamma'], 0.01
)
params['delta'] = st.sidebar.slider(
    "δ (интенсивность восстановления, 1/час):",
    0.1, 10.0, params['delta'], 0.1
)
params['alpha'] = st.sidebar.slider(
    "α (интенсивность перегрузки, 1/час):",
    0.01, 5.0, params['alpha'], 0.01
)
params['beta'] = st.sidebar.slider(
    "β (интенсивность восстановления буфера, 1/час):",
    0.1, 10.0, params['beta'], 0.1
)

# Основные расчеты
try:
    pi = calculate_pi(**params)
    
    # Отображение результатов
    st.subheader("Результаты")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Вероятность ожидания (π₀)", f"{pi[0]*100:.2f}%")
        st.metric("Вероятность перегрузки (π₃)", f"{pi[3]*100:.2f}%")
    with col2:
        st.metric("Вероятность обработки (π₁)", f"{pi[1]*100:.2f}%")
        st.metric("Коэффициент загрузки (ρ)", 
                 f"{params['lambda_']/params['mu']:.2f}")
    
    # График распределения вероятностей
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    states = ['Ожидание (π₀)', 'Обработка (π₁)', 'Сбой (π₂)', 'Перегрузка (π₃)']
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
    ax1.bar(states, pi, color=colors)
    ax1.set_ylabel("Вероятность")
    ax1.set_ylim(0, 1)
    st.pyplot(fig1)
    
    # Анализ чувствительности
    st.subheader("Анализ чувствительности")
    selected_param = st.selectbox(
        "Исследовать зависимость от:",
        ("λ (входящий поток)", "μ (обработка)", "γ (сбои)")
    )
    
    # Определяем диапазон значений для анализа
    if selected_param == "λ (входящий поток)":
        values = np.linspace(0.1, 30, 30)
        current_params = params.copy()
        param_key = 'lambda_'
    elif selected_param == "μ (обработка)":
        values = np.linspace(0.1, 30, 30)
        current_params = params.copy()
        param_key = 'mu'
    else:
        values = np.linspace(0.01, 5, 30)
        current_params = params.copy()
        param_key = 'gamma'
    
    # Расчет вероятностей перегрузки для разных значений параметра
    pi3_values = []
    for val in values:
        current_params[param_key] = val
        current_pi = calculate_pi(**current_params)
        pi3_values.append(current_pi[3])

# Построение графика чувствительности
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(values, pi3_values, 'o-', color='#9C27B0')
    ax2.set_xlabel(selected_param.split(' ')[0] + f" ({'1/час'})")
    ax2.set_ylabel("Вероятность перегрузки (π₃)")
    ax2.grid(True)
    st.pyplot(fig2)

except Exception as e:
    st.error(f"Ошибка в расчетах: {str(e)}")
    st.write("Текущие параметры:", params)

# Инструкция
st.markdown("---")
st.info("""
Инструкция:
1. Настройте параметры системы в боковой панели
2. Результаты обновляются автоматически
3. Используйте анализ чувствительности для исследования влияния параметров
""")

# Отладочная информация (можно скрыть)
with st.expander("Отладочная информация"):
    st.write("Матрица коэффициентов:")
    A_matrix = np.array([
        [-params['lambda_'], params['mu'], params['delta'], params['beta']],
        [params['lambda_'], -(params['mu'] + params['gamma'] + params['alpha']), 0, 0],
        [0, params['gamma'], -params['delta'], 0],
        [0, params['alpha'], 0, -params['beta']]
    ])
    st.write(A_matrix)
    st.write(f"Сумма вероятностей: {np.sum(pi):.6f}")