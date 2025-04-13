import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import simpy
from io import BytesIO
import sys
from importlib.metadata import version  # Для Python 3.12+

# Проверка версий
st.sidebar.markdown("### Версии библиотек")
try:
    st.sidebar.code(f"""
    Python: {sys.version.split()[0]}
    NumPy: {version('numpy')}
    Matplotlib: {version('matplotlib')}
    SimPy: {version('simpy')}
    Streamlit: {version('streamlit')}
    """)
except Exception as e:
    st.sidebar.error(f"Ошибка проверки версий: {e}")

# Модель системы
class SOISN_Model:
    def init(self, env, params):
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

# Интерфейс Streamlit
st.title("Моделирование СОИСН (Python 3.12+)")
st.markdown("""
Исследование установившегося режима системы обработки информации специального назначения
""")

# Параметры в сайдбаре
with st.sidebar:
    st.header("Параметры системы")
    params = {
        "lambda": st.slider("λ - Интенсивность входящего потока", 0.1, 30.0, 10.0),
        "mu": st.slider("μ - Интенсивность обработки", 0.1, 30.0, 15.0),
        "gamma": st.slider("γ - Интенсивность сбоев", 0.1, 5.0, 0.5),
        "delta": st.slider("δ - Интенсивность восстановления", 0.1, 5.0, 2.0),
        "alpha": st.slider("α - Интенсивность перегрузки", 0.1, 5.0, 1.0),
        "beta": st.slider("β - Интенсивность восстановления буфера", 0.1, 5.0, 3.0)
    }
    simulation_time = st.slider("Время моделирования (сек)", 100, 2000, 500)

# Запуск моделирования
if st.button("Запустить моделирование"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Инициализация модели...")
        env = simpy.Environment()
        model = SOISN_Model(env, params)
        env.process(model.run())
        
        status_text.text("Идет моделирование...")
        for i in range(simulation_time):
            env.run(until=i+1)
            progress_bar.progress((i+1)/simulation_time)
        
        # Расчет результатов
        total_time = max(1e-6, sum(model.time_in_states.values()))
        pi_sim = {state: t/total_time for state, t in model.time_in_states.items()}
        
        # Визуализация
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # График динамики состояний
        if model.state_history:
            times, states = zip(*model.state_history[:1000])  # Берем первые 1000 точек для производительности

state_codes = {"S0": 0, "S1": 1, "S2": 2, "S3": 3}
ax1.step(times, [state_codes[s] for s in states], where="post")
ax1.set_yticks([0, 1, 2, 3])
ax1.set_yticklabels(["Ожидание (S0)", "Обработка (S1)", "Сбой (S2)", "Перегрузка (S3)"])
ax1.set_xlabel("Время (сек)")
ax1.set_title("Динамика состояний")
ax1.grid(True)
        
        # График вероятностей
        states = list(pi_sim.keys())
        probs = list(pi_sim.values())
        colors = ["#4CAF50", "#2196F3", "#F44336", "#FF9800"]
        bars = ax2.bar(states, probs, color=colors)
        ax2.set_xlabel("Состояние")
        ax2.set_ylabel("Вероятность")
        ax2.set_title("Предельные вероятности")
        ax2.grid(True)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f"{height:.3f}", ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Вывод результатов
        st.success("Моделирование завершено!")
        st.pyplot(fig)
        
        # Таблица результатов
        st.subheader("Результаты")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("Параметры системы:")
            for param, value in params.items():
                st.text(f"{param}: {value:.2f}")
        
        with col2:
            st.markdown("Вероятности состояний:")
            for state, prob in pi_sim.items():
                st.text(f"{state}: {prob:.4f} ({prob*100:.1f}%)")
        
        # Коэффициент загрузки
        rho = params["lambda"] / params["mu"]
        st.metric("Коэффициент загрузки (ρ)", f"{rho:.2f}",
                help="ρ = λ/μ. При ρ > 1 система не справляется с нагрузкой")
        
 except Exception as e:
st.error(f"Ошибка моделирования: {type(e).name}: {str(e)}")
st.exception(e)
     finally:
        progress_bar.empty()

# Инструкция
with st.expander("Инструкция по использованию"):
    st.markdown("""
    1. Установите параметры системы в левой панели
    2. Нажмите кнопку "Запустить моделирование"
    3. Дождитесь завершения расчета
    4. Анализируйте результаты:
       - График динамики состояний
       - Вероятности каждого состояния
       - Коэффициент загрузки системы
    """)

# Дополнительная информация
st.caption("""
Дипломная работа Бирюкова Д.Р.  
Краснодарское высшее военное училище, 2025
""")