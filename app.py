import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Инициализация параметров
if 'params' not in st.session_state:
    st.session_state.params = {
        'lambda_': 10.0,
        'mu': 15.0,
        'gamma': 0.5,
        'delta': 2.0,
        'alpha': 1.0,
        'beta': 3.0
    }

# Правильная функция расчета
def calculate_pi(lambda_, mu, gamma, delta, alpha, beta):
    # Проверка входных параметров
    if mu == 0 or lambda_ < 0:
        raise ValueError("Некорректные параметры")
    
    # Матрица системы уравнений (исправленная версия)
    A = np.array([
        [-(lambda_ + mu), delta, beta, 0],
        [lambda_, -(mu + gamma + alpha), 0, 0],
        [0, gamma, -delta, 0],
        [0, alpha, 0, -beta]
    ])
    
    # Вектор правой части
    b = np.array([0, 0, 0, 0])
    
    try:
        # Решение системы
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Альтернативное решение если система вырождена
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Нормировка и проверка
    pi = np.abs(pi)
    pi /= pi.sum()
    return pi

# Интерфейс приложения
st.title("Модель СОИСН (рабочая версия)")

# Боковая панель параметров
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

# Основные расчеты и вывод
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
        st.metric("Коэффициент загрузки (ρ)", f"{params['lambda_']/params['mu']:.2f}")
    
    # График распределения вероятностей
    fig, ax = plt.subplots(figsize=(10, 5))
    states = ['Ожидание (π₀)', 'Обработка (π₁)', 'Сбой (π₂)', 'Перегрузка (π₃)']
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
    ax.bar(states, pi, color=colors)
    ax.set_ylabel("Вероятность")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

except Exception as e:
    st.error(f"Ошибка в расчетах: {str(e)}")
    st.write("Проверьте введенные параметры")

# Тестовые сценарии
st.sidebar.markdown("---")
st.sidebar.subheader("Тестовые данные")
test_cases = {
    "Сбалансированная": {'lambda_':10, 'mu':15, 'gamma':0.5, 'delta':2, 'alpha':1, 'beta':3},
    "Перегруженная": {'lambda_':25, 'mu':10, 'gamma':1, 'delta':3, 'alpha':2, 'beta':5},
    "Недогруженная": {'lambda_':5, 'mu':20, 'gamma':0.1, 'delta':1, 'alpha':0.5, 'beta':2}
}

for name, values in test_cases.items():
    if st.sidebar.button(name):
        st.session_state.params.update(values)
        st.rerun()

# Отладочная информация
with st.expander("Техническая информация"):
    st.write("Текущие параметры:", params)
    try:
        A = np.array([
            [-(params['lambda_'] + params['mu']), params['delta'], params['beta'], 0],
            [params['lambda_'], -(params['mu'] + params['gamma'] + params['alpha']), 0, 0],
            [0, params['gamma'], -params['delta'], 0],
            [0, params['alpha'], 0, -params['beta']]
        ])
        st.write("Матрица системы:", A)
        st.write("Собственные значения:", np.linalg.eigvals(A))
    except:
        pass