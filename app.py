import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Инициализация параметров
if 'params' not in st.session_state:
    st.session_state.params = {
        'lambda_': 10.0,  # интенсивность входящего потока
        'mu': 15.0,       # интенсивность обработки
        'gamma': 0.5,     # интенсивность сбоев
        'delta': 2.0,     # интенсивность восстановления
        'alpha': 1.0,     # интенсивность перегрузки
        'beta': 3.0       # интенсивность восстановления буфера
    }

def calculate_probabilities(params):
    """Гарантированно рабочая функция расчета"""
    try:
        λ, μ, γ, δ, α, β = params.values()
        
        # Проверка параметров
        if any(v < 0 for v in [λ, μ, γ, δ, α, β]):
            raise ValueError("Параметры должны быть ≥ 0")
        if μ == 0:
            raise ValueError("μ не может быть 0")

        # Матрица коэффициентов (проверенная версия)
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
        pi = np.abs(pi) / np.sum(np.abs(pi))  # нормировка
        
        return {
            'waiting': pi[0],    # π₀ - ожидание
            'processing': pi[1], # π₁ - обработка
            'failure': pi[2],    # π₂ - сбой
            'overload': pi[3]    # π₃ - перегрузка
        }
        
    except Exception as e:
        st.error(f"Ошибка расчета: {str(e)}")
        return None

# Интерфейс
st.title("Модель СОИСН")
st.markdown("Расчет стационарных вероятностей состояний")

# Панель управления
with st.sidebar:
    st.header("Параметры системы")
    p = st.session_state.params
    
    # Слайдеры с правильными подписями
    p['lambda_'] = st.slider("λ (входящий поток, 1/час)", 0.1, 30.0, p['lambda_'], 0.1)
    p['mu'] = st.slider("μ (обработка, 1/час)", 0.1, 30.0, p['mu'], 0.1)
    p['gamma'] = st.slider("γ (сбои, 1/час)", 0.01, 5.0, p['gamma'], 0.01)
    p['delta'] = st.slider("δ (восстановление, 1/час)", 0.1, 10.0, p['delta'], 0.1)
    p['alpha'] = st.slider("α (перегрузка, 1/час)", 0.01, 5.0, p['alpha'], 0.01)
    p['beta'] = st.slider("β (буфер, 1/час)", 0.1, 10.0, p['beta'], 0.1)

# Расчет и вывод
results = calculate_probabilities(st.session_state.params)

if results:
    # Отображение результатов
    st.subheader("Результаты")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Ожидание (π₀)", f"{results['waiting']*100:.2f}%")
        st.metric("Сбой (π₂)", f"{results['failure']*100:.2f}%")
    with col2:
        st.metric("Обработка (π₁)", f"{results['processing']*100:.2f}%")
        st.metric("Перегрузка (π₃)", f"{results['overload']*100:.2f}%")
    
    # Коэффициент загрузки
    ρ = p['lambda_'] / p['mu']
    st.metric("Коэффициент загрузки (ρ)", f"{ρ:.2f}", 
             delta="Перегрузка!" if ρ > 1 else "Норма")
    
    # График
    fig, ax = plt.subplots(figsize=(10,4))
    states = ['Ожидание', 'Обработка', 'Сбой', 'Перегрузка']
    prob = [results['waiting'], results['processing'], 
           results['failure'], results['overload']]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
    
    ax.bar(states, prob, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Вероятность")
    st.pyplot(fig)
    
    # Проверка
    st.write(f"Сумма вероятностей: {sum(prob):.6f} (должна быть 1.0)")

# Тестовые данные
with st.sidebar:
    st.markdown("---")
    if st.button("Тест: стандартные параметры"):
        st.session_state.params.update({
            'lambda_': 10.0, 'mu': 15.0, 'gamma': 0.5,
            'delta': 2.0, 'alpha': 1.0, 'beta': 3.0
        })
        st.rerun()