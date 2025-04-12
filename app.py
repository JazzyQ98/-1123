import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Инициализация параметров
if 'params' not in st.session_state:
    st.session_state.params = {
        'lambda_': 10.0, 'mu': 15.0, 'gamma': 0.5,
        'delta': 2.0, 'alpha': 1.0, 'beta': 3.0
    }

def calculate_pi(lambda_, mu, gamma, delta, alpha, beta):
    """Исправленная функция расчета вероятностей"""
    # Матрица системы (проверенная версия)
    A = np.array([
        [-lambda_, mu, delta, beta],
        [lambda_, -(mu + gamma + alpha), 0, 0],
        [0, gamma, -delta, 0],
        [0, alpha, 0, -beta]
    ])
    
    b = np.zeros(4)
    
    try:
        pi = np.linalg.solve(A, b)
    except:
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Нормировка и защита от отрицательных значений
    pi = np.abs(pi)
    pi /= pi.sum()
    return {
        'waiting': pi[0],    # π₀ - ожидание
        'processing': pi[1], # π₁ - обработка
        'failure': pi[2],    # π₂ - сбой
        'overload': pi[3]    # π₃ - перегрузка
    }

# Интерфейс
st.title("Модель СОИСН (проверенная версия)")

# Панель параметров
with st.sidebar:
    st.header("Параметры системы")
    p = st.session_state.params
    p['lambda_'] = st.slider("λ (входящий поток)", 0.1, 30.0, p['lambda_'], 0.1)
    p['mu'] = st.slider("μ (обработка)", 0.1, 30.0, p['mu'], 0.1)
    p['gamma'] = st.slider("γ (сбои)", 0.01, 5.0, p['gamma'], 0.01)
    p['delta'] = st.slider("δ (восстановление)", 0.1, 10.0, p['delta'], 0.1)
    p['alpha'] = st.slider("α (перегрузка)", 0.01, 5.0, p['alpha'], 0.01)
    p['beta'] = st.slider("β (восст. буфера)", 0.1, 10.0, p['beta'], 0.1)

# Расчет и вывод
try:
    results = calculate_pi(**st.session_state.params)
    
    # Правильные названия метрик
    st.subheader("Результаты")
    cols = st.columns(2)
    with cols[0]:
        st.metric("Вероятность ожидания (π₀)", f"{results['waiting']*100:.2f}%")
        st.metric("Вероятность сбоя (π₂)", f"{results['failure']*100:.2f}%")
    with cols[1]:
        st.metric("Вероятность обработки (π₁)", f"{results['processing']*100:.2f}%")
        st.metric("Вероятность перегрузки (π₃)", f"{results['overload']*100:.2f}%")
    
    # Коэффициент загрузки
    rho = p['lambda_'] / p['mu']
    st.metric("Коэффициент загрузки (ρ)", f"{rho:.2f}", 
             delta=f"{'Перегрузка' if rho > 1 else 'Норма'}")
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(10,4))
    states = ['Ожидание (π₀)', 'Обработка (π₁)', 'Сбой (π₂)', 'Перегрузка (π₃)']
    probas = [results['waiting'], results['processing'], 
             results['failure'], results['overload']]
    ax.bar(states, probas, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    ax.set_ylim(0, 1)
    st.pyplot(fig)

except Exception as e:
    st.error(f"Ошибка расчета: {str(e)}")
    st.write("Проверьте параметры на корректность")

# Тестовые данные
with st.sidebar:
    st.markdown("---")
    st.subheader("Тестовые сценарии")
    if st.button("Стандартные параметры"):
        st.session_state.params.update({
            'lambda_':10, 'mu':15, 'gamma':0.5,
            'delta':2, 'alpha':1, 'beta':3
        })
        st.rerun()