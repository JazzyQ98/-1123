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
        [-lambda_, mu,      delta,  beta],
        [lambda_, -(mu + gamma + alpha), 0, 0],
        [0,       gamma,    -delta, 0],
        [0,       alpha,    0,      -beta],
        [1,       1,        1,      1]
    ])
    
    # Решаем первые 4 уравнения
    try:
        pi = np.linalg.solve(A[:4,:4], [0,0,0,0])
    except np.linalg.LinAlgError:
        pi = np.linalg.lstsq(A[:4,:4], [0,0,0,0], rcond=None)[0]
    
    pi = np.append(pi, 0)  # Добавляем нулевую компоненту
    pi /= pi.sum()         # Нормировка
    return pi[:4]          # Возвращаем 4 вероятности
st.write("Проверка матрицы A:")
st.write(A)
st.write("Сумма вероятностей:", sum(pi))
# Заголовок приложения
st.title("Модель установившегося режима СОИСН")
st.markdown("""
Анализ вероятностей состояний системы обработки информации специального назначения.
""")

# Боковая панель для ввода параметров
st.sidebar.header("Параметры системы")

# Обновляем параметры в session_state при изменении ползунков
st.session_state.params['lambda_'] = st.sidebar.slider(
    "λ (интенсивность входящего потока):", 
    1.0, 30.0, st.session_state.params['lambda_'], 0.1
)
st.session_state.params['mu'] = st.sidebar.slider(
    "μ (интенсивность обработки):", 
    1.0, 30.0, st.session_state.params['mu'], 0.1
)
st.session_state.params['gamma'] = st.sidebar.slider(
    "γ (интенсивность сбоев):", 
    0.1, 5.0, st.session_state.params['gamma'], 0.1
)
st.session_state.params['delta'] = st.sidebar.slider(
    "δ (интенсивность восстановления):", 
    0.1, 10.0, st.session_state.params['delta'], 0.1
)
st.session_state.params['alpha'] = st.sidebar.slider(
    "α (интенсивность перегрузки):", 
    0.1, 5.0, st.session_state.params['alpha'], 0.1
)
st.session_state.params['beta'] = st.sidebar.slider(
    "β (интенсивность восстановления буфера):", 
    0.1, 10.0, st.session_state.params['beta'], 0.1
)

# Расчет вероятностей с текущими параметрами
try:
    pi = calculate_pi(**st.session_state.params)
    
    # Вывод результатов
    st.subheader("Результаты")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Вероятность перегрузки (π₃)", f"{pi[3] * 100:.2f}%")
        st.metric("Коэффициент загрузки (ρ)", f"{st.session_state.params['lambda_'] / st.session_state.params['mu']:.2f}")
    with col2:
        st.metric("Вероятность ожидания (π₀)", f"{pi[0] * 100:.2f}%")
        st.metric("Вероятность обработки (π₁)", f"{pi[1] * 100:.2f}%")

    # График вероятностей
    fig, ax = plt.subplots(figsize=(8, 4))
    states = ['Ожидание (π₀)', 'Обработка (π₁)', 'Сбой (π₂)', 'Перегрузка (π₃)']
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
    ax.bar(states, pi, color=colors)
    ax.set_ylabel("Вероятность")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    # Анализ чувствительности
    st.subheader("Анализ чувствительности")
    selected_param = st.selectbox(
        "Исследовать зависимость от:",
        ("λ (входящий поток)", "μ (обработка)", "γ (сбои)")
    )

    if selected_param == "λ (входящий поток)":
        values = np.linspace(1, 30, 20)
        param_name = "λ"
    elif selected_param == "μ (обработка)":
        values = np.linspace(1, 30, 20)
        param_name = "μ"
    else:
        values = np.linspace(0.1, 5, 20)
        param_name = "γ"

    pi3_values = []
    for val in values:
        if param_name == "λ":
            current_pi = calculate_pi(val, 
                                   st.session_state.params['mu'],
                                   st.session_state.params['gamma'],
                                   st.session_state.params['delta'],
                                   st.session_state.params['alpha'],
                                   st.session_state.params['beta'])
        elif param_name == "μ":
            current_pi = calculate_pi(st.session_state.params['lambda_'],
val,
                                   st.session_state.params['gamma'],
                                   st.session_state.params['delta'],
                                   st.session_state.params['alpha'],
                                   st.session_state.params['beta'])
        else:
            current_pi = calculate_pi(st.session_state.params['lambda_'],
                                   st.session_state.params['mu'],
                                   val,
                                   st.session_state.params['delta'],
                                   st.session_state.params['alpha'],
                                   st.session_state.params['beta'])
        pi3_values.append(current_pi[3])

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(values, pi3_values, marker='o', color='#9C27B0')
    ax2.set_xlabel(selected_param)
    ax2.set_ylabel("Вероятность перегрузки (π₃)")
    st.pyplot(fig2)

except Exception as e:
    st.error(f"Ошибка в расчетах: {str(e)}")

# Инструкция
st.markdown("---")
st.info("""
Инструкция:
1. Настройте параметры системы в боковой панели.
2. Результаты автоматически обновляются при изменении параметров.
3. Используйте анализ чувствительности для исследования влияния параметров.
""")