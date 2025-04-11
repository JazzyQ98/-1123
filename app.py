import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Функция для расчета вероятностей
def calculate_pi(lambda_, mu, gamma, delta, alpha, beta):
    # Корректная матрица коэффициентов
    A = np.array([
        [-lambda_, mu,      delta,  beta],      # Уравнение для π₀
        [lambda_, -(mu + gamma + alpha), 0, 0],  # Уравнение для π₁
        [0,       gamma,    -delta, 0],         # Уравнение для π₂
        [0,       alpha,    0,      -beta],     # Уравнение для π₃
        [1,       1,        1,      1]          # Условие нормировки
    ])
    
    b = np.array([0, 0, 0, 0, 1])  # Вектор правой части
    
    # Решение системы с проверкой
    try:
        # Используем точный метод решения
        pi = np.linalg.solve(A[:4,:4], [0, 0, 0, 0])
        pi = np.append(pi, 0)  # Добавляем нулевую компоненту
        pi /= pi.sum()         # Нормировка
        return pi[:4]          # Возвращаем только нужные 4 компоненты
        
    except np.linalg.LinAlgError:
        # Если система вырождена, используем МНК
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
        pi /= pi.sum()
        return pi[:4]

# Интерфейс приложения
st.title("Модель установившегося режима СОИСН")

# Боковая панель с параметрами
st.sidebar.header("Параметры системы")
lambda_ = st.sidebar.slider("λ (интенсивность входящего потока, 1/час):", 
                          0.1, 30.0, 10.0, 0.1)
mu = st.sidebar.slider("μ (интенсивность обработки, 1/час):", 
                     0.1, 30.0, 15.0, 0.1)
gamma = st.sidebar.slider("γ (интенсивность сбоев, 1/час):", 
                       0.01, 5.0, 0.5, 0.01)
delta = st.sidebar.slider("δ (интенсивность восстановления, 1/час):", 
                       0.1, 10.0, 2.0, 0.1)
alpha = st.sidebar.slider("α (интенсивность перегрузки, 1/час):", 
                       0.01, 5.0, 1.0, 0.01)
beta = st.sidebar.slider("β (интенсивность восстановления буфера, 1/час):", 
                      0.1, 10.0, 3.0, 0.1)

# Расчет вероятностей
pi = calculate_pi(lambda_, mu, gamma, delta, alpha, beta)

# Отображение результатов
st.subheader("Результаты")
col1, col2 = st.columns(2)

with col1:
    st.metric("Вероятность ожидания (π₀)", f"{pi[0]*100:.2f}%")
    st.metric("Вероятность обработки (π₁)", f"{pi[1]*100:.2f}%")
    
with col2:
    st.metric("Вероятность сбоя (π₂)", f"{pi[2]*100:.2f}%")
    st.metric("Вероятность перегрузки (π₃)", f"{pi[3]*100:.2f}%")

st.metric("Коэффициент загрузки (ρ)", f"{lambda_/mu:.2f}")

# Визуализация
fig, ax = plt.subplots(figsize=(10, 5))
states = ['Ожидание (π₀)', 'Обработка (π₁)', 'Сбой (π₂)', 'Перегрузка (π₃)']
colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
ax.bar(states, pi, color=colors)
ax.set_ylabel("Вероятность")
ax.set_ylim(0, 1)
st.pyplot(fig)

# Отладочная информация (можно скрыть)
with st.expander("Техническая информация"):
    st.write("Матрица системы:")
    A_matrix = np.array([
        [-lambda_, mu, delta, beta],
        [lambda_, -(mu + gamma + alpha), 0, 0],
        [0, gamma, -delta, 0],
        [0, alpha, 0, -beta]
    ])
    st.write(A_matrix)
    st.write(f"Сумма вероятностей: {pi.sum():.6f}")
    st.write(f"Собственные значения: {np.linalg.eigvals(A_matrix)}")