import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Заголовок приложения
st.title("Модель установившегося режима СОИСН")
st.markdown("""
Анализ вероятностей состояний системы обработки информации специального назначения.
""")

# Боковая панель для ввода параметров
st.sidebar.header("Параметры системы")
lambda_ = st.sidebar.slider("λ (интенсивность входящего потока):", 1.0, 30.0, 10.0, 0.1)
mu = st.sidebar.slider("μ (интенсивность обработки):", 1.0, 30.0, 15.0, 0.1)
gamma = st.sidebar.slider("γ (интенсивность сбоев):", 0.1, 5.0, 0.5, 0.1)
delta = st.sidebar.slider("δ (интенсивность восстановления):", 0.1, 10.0, 2.0, 0.1)
alpha = st.sidebar.slider("α (интенсивность перегрузки):", 0.1, 5.0, 1.0, 0.1)
beta = st.sidebar.slider("β (интенсивность восстановления буфера):", 0.1, 10.0, 3.0, 0.1)

# Расчет предельных вероятностей
def calculate_pi(lambda_, mu, gamma, delta, alpha, beta):
    A = np.array([
        [-lambda_, mu, delta, beta],
        [lambda_, -(mu + gamma + alpha), 0, 0],
        [0, gamma, -delta, 0],
        [0, alpha, 0, -beta],
        [1, 1, 1, 1]
    ])
    b = np.array([0, 0, 0, 0, 1])
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    return pi

pi = calculate_pi(lambda_, mu, gamma, delta, alpha, beta)

# Вывод результатов
st.subheader("Результаты")
col1, col2 = st.columns(2)
with col1:
    st.metric("Вероятность перегрузки (π₃)", f"{pi[3] * 100:.2f}%")
    st.metric("Коэффициент загрузки (ρ)", f"{lambda_ / mu:.2f}")
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
        pi = calculate_pi(val, mu, gamma, delta, alpha, beta)
    elif param_name == "μ":
        pi = calculate_pi(lambda_, val, gamma, delta, alpha, beta)
    else:
        pi = calculate_pi(lambda_, mu, val, delta, alpha, beta)
    pi3_values.append(pi[3])

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(values, pi3_values, marker='o', color='#9C27B0')
ax2.set_xlabel(selected_param)
ax2.set_ylabel("Вероятность перегрузки (π₃)")
st.pyplot(fig2)

# Инструкция
st.markdown("---")
st.info("""
Инструкция:
1. Настройте параметры системы в боковой панели.
2. Результаты автоматически обновляются.
3. Используйте анализ чувствительности для исследования влияния параметров.
""")