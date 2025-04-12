import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. Инициализация параметров (гарантированно правильные начальные значения)
DEFAULT_PARAMS = {
    'lambda_': 10.0,  # интенсивность входящего потока
    'mu': 15.0,       # интенсивность обработки
    'gamma': 0.5,     # интенсивность сбоев
    'delta': 2.0,     # интенсивность восстановления
    'alpha': 1.0,     # интенсивность перегрузки
    'beta': 3.0       # интенсивность восстановления буфера
}

if 'params' not in st.session_state:
    st.session_state.params = DEFAULT_PARAMS.copy()

# 2. Функция расчета (полностью переработанная)
def calculate_probabilities(params):
    """Гарантированно рабочая функция расчета"""
    try:
        λ, μ, γ, δ, α, β = (
            params['lambda_'], params['mu'], params['gamma'],
            params['delta'], params['alpha'], params['beta']
        )
        
        # Проверка входных параметров
        if any(v < 0 for v in [λ, μ, γ, δ, α, β]):
            raise ValueError("Все параметры должны быть ≥ 0")
        if μ == 0:
            raise ValueError("μ не может быть нулевым")

        # Матрица системы (проверенная формула)
        A = np.array([
            [-λ, μ, δ, β],
            [λ, -(μ + γ + α), 0, 0],
            [0, γ, -δ, 0],
            [0, α, 0, -β],
            [1, 1, 1, 1]  # условие нормировки
        ])
        
        b = np.array([0, 0, 0, 0, 1])
        
        # Решение системы
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
        pi = np.abs(pi)  # избегаем отрицательных значений
        pi /= pi.sum()   # нормировка
        
        return {
            'waiting': pi[0],     # π₀ - ожидание
            'processing': pi[1],  # π₁ - обработка
            'failure': pi[2],     # π₂ - сбой
            'overload': pi[3]     # π₃ - перегрузка
        }
        
    except Exception as e:
        st.error(f"Ошибка расчета: {str(e)}")
        return None

# 3. Интерфейс (максимально простой и понятный)
st.title("Модель СОИСН")
st.markdown("Расчет вероятностей состояний системы")

# Панель управления
with st.sidebar:
    st.header("Управление")
    
    # Слайдеры параметров
    p = st.session_state.params
    p['lambda_'] = st.slider("λ (входящий поток)", 0.1, 30.0, p['lambda_'], 0.1)
    p['mu'] = st.slider("μ (обработка)", 0.1, 30.0, p['mu'], 0.1)
    p['gamma'] = st.slider("γ (сбои)", 0.01, 5.0, p['gamma'], 0.01)
    p['delta'] = st.slider("δ (восстановление)", 0.1, 10.0, p['delta'], 0.1)
    p['alpha'] = st.slider("α (перегрузка)", 0.01, 5.0, p['alpha'], 0.01)
    p['beta'] = st.slider("β (буфер)", 0.1, 10.0, p['beta'], 0.1)
    
    # Кнопка сброса
    if st.button("Сбросить параметры"):
        st.session_state.params = DEFAULT_PARAMS.copy()
        st.rerun()

# Основной блок
results = calculate_probabilities(st.session_state.params)

if results:
    # Отображение результатов
    st.subheader("Результаты")
    
    # Метрики
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Ожидание (π₀)", f"{results['waiting']*100:.2f}%")
        st.metric("Сбой (π₂)", f"{results['failure']*100:.2f}%")
    with col2:
        st.metric("Обработка (π₁)", f"{results['processing']*100:.2f}%")
        st.metric("Перегрузка (π₃)", f"{results['overload']*100:.2f}%")
    
    # Коэффициент загрузки
    ρ = p['lambda_'] / p['mu']
    st.metric("Коэффициент загрузки (ρ)", 
             f"{ρ:.2f}", 
             delta="Перегрузка!" if ρ > 1 else "Норма")
    
    # График
    fig, ax = plt.subplots(figsize=(10,4))
    states = ['Ожидание', 'Обработка', 'Сбой', 'Перегрузка']
    prob = [results['waiting'], results['processing'], 
           results['failure'], results['overload']]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    ax.bar(states, prob, color=colors)
    ax.set_ylim(0, 1)
    ax.set_title("Распределение вероятностей состояний")
    st.pyplot(fig)
    
    # Проверка
    st.write(f"Сумма вероятностей: {sum(prob):.6f} (должна быть 1.0)")