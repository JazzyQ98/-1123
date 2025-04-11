import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Инициализация параметров
if 'params' not in st.session_state:
    st.session_state.params = {
        'lambda_': 10.0, 'mu': 15.0, 'gamma': 0.5,
        'delta': 2.0, 'alpha': 1.0, 'beta': 3.0
    }

# Правильная функция расчета (исправленная)
def calculate_pi(lambda_, mu, gamma, delta, alpha, beta):
    # Матрица системы уравнений
    A = np.array([
        [-lambda_, mu, 0, 0],          # Уравнение для π₀
        [lambda_, -(mu + gamma + alpha), delta, beta],  # Уравнение для π₁
        [0, gamma, -delta, 0],         # Уравнение для π₂
        [0, alpha, 0, -beta]           # Уравнение для π₃
    ])
    
    b = np.array([0, 0, 0, 0])
    
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Нормировка и проверка
    pi = np.abs(pi) / np.sum(np.abs(pi))
    return pi

# Интерфейс
st.title("Модель СОИСН (исправленная)")

# Слайдеры параметров
params = st.session_state.params
params['lambda_'] = st.sidebar.slider("λ (входящий поток)", 0.1, 30.0, params['lambda_'], 0.1)
params['mu'] = st.sidebar.slider("μ (обработка)", 0.1, 30.0, params['mu'], 0.1)
params['gamma'] = st.sidebar.slider("γ (сбои)", 0.01, 5.0, params['gamma'], 0.01)
params['delta'] = st.sidebar.slider("δ (восстановление)", 0.1, 10.0, params['delta'], 0.1)
params['alpha'] = st.sidebar.slider("α (перегрузка)", 0.01, 5.0, params['alpha'], 0.01)
params['beta'] = st.sidebar.slider("β (восст. буфера)", 0.1, 10.0, params['beta'], 0.1)

# Расчет и вывод
try:
    pi = calculate_pi(**params)
    
    st.subheader("Результаты")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("π₀ (ожидание)", f"{pi[0]*100:.2f}%")
        st.metric("π₃ (перегрузка)", f"{pi[3]*100:.2f}%")
    with col2:
        st.metric("π₁ (обработка)", f"{pi[1]*100:.2f}%")
        st.metric("ρ (коэф. загрузки)", f"{params['lambda_']/params['mu']:.2f}")
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(['π₀', 'π₁', 'π₂', 'π₃'], pi, color=['green', 'blue', 'orange', 'red'])
    ax.set_ylim(0, 1)
    st.pyplot(fig)

except Exception as e:
    st.error(f"Ошибка: {str(e)}")

# Тестовые сценарии
st.sidebar.markdown("---")
st.sidebar.subheader("Тестовые данные")
if st.sidebar.button("Сбалансированная система"):
    st.session_state.params.update({'lambda_':10, 'mu':15, 'gamma':0.5, 'delta':2, 'alpha':1, 'beta':3})
    st.rerun()

if st.sidebar.button("Перегруженная система"):
    st.session_state.params.update({'lambda_':25, 'mu':10, 'gamma':1, 'delta':3, 'alpha':2, 'beta':5})
    st.rerun()

if st.sidebar.button("Недогруженная система"):
    st.session_state.params.update({'lambda_':5, 'mu':20, 'gamma':0.1, 'delta':1, 'alpha':0.5, 'beta':2})
    st.rerun()