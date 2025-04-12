import numpy as np
import matplotlib.pyplot as plt
import simpy
import streamlit as st
from io import BytesIO
st.write(f"Python: {sys.version}")
st.write(f"SimPy version: {simpy.version}")

# Настройка страницы
st.set_page_config(layout="wide", page_title="Моделирование СОИСН")
st.title("Имитационная модель системы обработки информации специального назначения")
st.markdown("""
Дипломная работа Бирюкова Д.Р.  
*Исследование установившегося режима функционирования СОИСН*
""")

# Боковая панель с параметрами
with st.sidebar:
    st.header("Параметры системы")
    simulation_time = st.slider("Время моделирования (сек)", 100, 5000, 1000)
    lambda_ = st.slider("λ - Интенсивность входящего потока", 1.0, 30.0, 10.0)
    mu = st.slider("μ - Интенсивность обработки", 1.0, 30.0, 15.0)
    gamma = st.slider("γ - Интенсивность сбоев", 0.1, 5.0, 0.5)
    delta = st.slider("δ - Интенсивность восстановления", 0.1, 5.0, 2.0)
    alpha = st.slider("α - Интенсивность перегрузки", 0.1, 5.0, 1.0)
    beta = st.slider("β - Интенсивность восстановления буфера", 0.1, 5.0, 3.0)

    params = {
        "lambda": lambda_,
        "mu": mu,
        "gamma": gamma,
        "delta": delta,
        "alpha": alpha,
        "beta": beta
    }

# Класс модели
class SOISN_Model:
    def init(self, env, params):
        if not isinstance(env, simpy.Environment):
            raise TypeError("env должен быть объектом simpy.Environment")
        if not isinstance(params, dict):
            raise TypeError("params должен быть словарем")
        
        self.env = env
        self.params = params
        self.state = "S0"
        self.state_history = []
        self.time_in_states = {"S0": 0, "S1": 0, "S2": 0, "S3": 0}
        
    def run(self):
        while True:
            start_time = self.env.now
            self.state_history.append((self.env.now, self.state))
            
            if self.state == "S0":
                yield self.env.timeout(np.random.exponential(1/self.params["lambda"]))
                self.state = "S1"
                
            elif self.state == "S1":
                rates = {
                    "S0": self.params["mu"],
                    "S2": self.params["gamma"],
                    "S3": self.params["alpha"]
                }
                total_rate = sum(rates.values())
                next_state = np.random.choice(
                    list(rates.keys()),
                    p=[r/total_rate for r in rates.values()]
                )
                yield self.env.timeout(np.random.exponential(1/total_rate))
                self.state = next_state
                
            elif self.state == "S2":
                yield self.env.timeout(np.random.exponential(1/self.params["delta"]))
                self.state = "S0"
                
            elif self.state == "S3":
                yield self.env.timeout(np.random.exponential(1/self.params["beta"]))
                self.state = "S0"
            
            self.time_in_states[self.state_history[-1][1]] += self.env.now - start_time

# Функция для аналитического расчета
def calculate_analytic(params):
    try:
        lambda_ = params["lambda"]
        mu = params["mu"]
        gamma = params["gamma"]
        delta = params["delta"]
        alpha = params["alpha"]
        beta = params["beta"]
        
        # Проверка деления на ноль
        denom = mu + gamma + alpha
        if denom == 0:
            raise ValueError("Знаменатель не может быть нулевым")
        
        pi_1 = lambda_ / denom
        pi_2 = (gamma / delta) * pi_1 if delta != 0 else 0
        pi_3 = (alpha / beta) * pi_1 if beta != 0 else 0
        
        pi_0 = 1 / (1 + pi_1 + pi_2 + pi_3)
        
        return {
            "S0": pi_0,
            "S1": pi_1 * pi_0,
            "S2": pi_2 * pi_0,
            "S3": pi_3 * pi_0
        }
    except Exception as e:
        st.error(f"Ошибка в calculate_analytic(): {str(e)}")
        return {"S0": 0, "S1": 0, "S2": 0, "S3": 0}  # Возвращаем нули при ошибке

# Запуск моделирования
def run_simulation():
    env = simpy.Environment()
    model = SOISN_Model(env, params)
    env.process(model.run())
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(simulation_time):
        env.run(until=i+1)
        progress = (i+1)/simulation_time
        progress_bar.progress(progress)
        status_text.text(f"Прогресс: {int(progress*100)}%")
    
    total_time = sum(model.time_in_states.values())
    pi_sim = {state: t/total_time for state, t in model.time_in_states.items()}
    pi_analytic = calculate_analytic(params)
    
    return model, pi_sim, pi_analytic

# Визуализация
def plot_results(model, pi_sim, pi_analytic):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # График состояний
    times, states = zip(*model.state_history)
    state_codes = {"S0": 0, "S1": 1, "S2": 2, "S3": 3}
    coded_states = [state_codes[s] for s in states]
    
    ax1.step(times, coded_states, where="post")
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(["S0 (Ожидание)", "S1 (Обработка)", "S2 (Сбой)", "S3 (Перегрузка)"])
    ax1.set_xlabel("Время (сек)")
    ax1.set_title("Динамика состояний системы")
    ax1.grid(True)
    
    # График вероятностей
    states = list(pi_sim.keys())
    sim_probs = list(pi_sim.values())
    analytic_probs = [pi_analytic[s] for s in states]
    
    x = np.arange(len(states))
    width = 0.35
    
    ax2.bar(x - width/2, sim_probs, width, label='Имитация', color='royalblue')
    ax2.bar(x + width/2, analytic_probs, width, label='Аналитика', color='orange')
    
    for i, (sim, ana) in enumerate(zip(sim_probs, analytic_probs)):
        ax2.text(i - width/2, sim + 0.01, f"{sim:.3f}", ha='center')
        ax2.text(i + width/2, ana + 0.01, f"{ana:.3f}", ha='center')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(states)
    ax2.set_ylabel("Вероятность")
    ax2.set_title("Сравнение вероятностей")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

# Основной блок
if st.button("Запустить моделирование"):
    with st.spinner("Идет моделирование..."):
        model, pi_sim, pi_analytic = run_simulation()
    
    st.success("Моделирование завершено!")
    
    # Вывод результатов
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Параметры системы")
        st.write(f"λ (входящий поток): {params['lambda']:.1f}")
        st.write(f"μ (обработка): {params['mu']:.1f}")
        st.write(f"γ (сбои): {params['gamma']:.1f}")
        st.write(f"δ (восстановление): {params['delta']:.1f}")
        st.write(f"α (перегрузка): {params['alpha']:.1f}")
        st.write(f"β (восстановление буфера): {params['beta']:.1f}")
    
    with col2:
        st.subheader("Результаты")
        st.write("Имитационные вероятности:")
        for state, prob in pi_sim.items():
            st.write(f"- {state}: {prob:.4f} ({prob*100:.1f}%)")
        
        st.write("Аналитические вероятности:")
        for state, prob in pi_analytic.items():
            st.write(f"- {state}: {prob:.4f} ({prob*100:.1f}%)")
    
    # Графики
    st.subheader("Визуализация результатов")
    fig = plot_results(model, pi_sim, pi_analytic)
    st.pyplot(fig)
    
    # Коэффициент загрузки
    rho = params['lambda'] / params['mu']
    st.metric("Коэффициент загрузки (ρ)", f"{rho:.2f}", 
              help="ρ = λ/μ. При ρ > 1 система не справляется с нагрузкой")
    
    # Экспорт результатов
    st.subheader("Экспорт результатов")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    st.download_button(
        label="Скачать графики",
        data=buf.getvalue(),
        file_name="soisn_results.png",
        mime="image/png"
    )
    
    # Отчет
    report = f"""
    Отчет по моделированию СОИСН
    ============================
    Параметры:
    - λ = {params['lambda']:.1f}
    - μ = {params['mu']:.1f}
    - γ = {params['gamma']:.1f}
    - δ = {params['delta']:.1f}
    - α = {params['alpha']:.1f}
    - β = {params['beta']:.1f}
    
    Результаты:
    - Коэффициент загрузки ρ = {rho:.2f}
    
    Вероятности состояний:
    | Состояние | Имитация | Аналитика |
    |-----------|----------|-----------|
    | S0        | {pi_sim['S0']:.4f} | {pi_analytic['S0']:.4f} |
    | S1        | {pi_sim['S1']:.4f} | {pi_analytic['S1']:.4f} |
    | S2        | {pi_sim['S2']:.4f} | {pi_analytic['S2']:.4f} |
    | S3        | {pi_sim['S3']:.4f} | {pi_analytic['S3']:.4f} |
    """
    
    st.download_button(
        label="Скачать отчет (TXT)",
        data=report.encode("utf-8"),
        file_name="soisn_report.txt",
        mime="text/plain"
    )

# Описание модели
with st.expander("Описание модели"):
    st.markdown("""
    ### Модель системы обработки информации специального назначения (СОИСН)
    
    Состояния системы:
    1. S0 (Ожидание) - система готова к обработке сообщений
    2. S1 (Обработка) - активная обработка сообщения
    3. S2 (Сбой) - восстановление после отказа
    4. S3 (Перегрузка) - буфер переполнен, требуется восстановление
    
    Параметры переходов:
    - λ - интенсивность входящего потока сообщений
    - μ - интенсивность обработки сообщений
    - γ - интенсивность возникновения сбоев
    - δ - интенсивность восстановления после сбоя
    - α - интенсивность перехода в состояние перегрузки
    - β - интенсивность восстановления буфера
    
    Теоретическая основа:
    - Марковский процесс с непрерывным временем
    - Уравнения Колмогорова для предельных вероятностей
    """)