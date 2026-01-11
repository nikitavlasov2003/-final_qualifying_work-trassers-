import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# НАСТРОЙКИ СТРАНИЦЫ 
st.set_page_config(page_title="Well Classifier", page_icon="🛢️", layout="wide")

# ЗАГРУЗКА ДАННЫХ
@st.cache_resource
def load_assets():
    try:
        data = {
            'model1': joblib.load('models/first_model_3class.pkl'),
            'model2': joblib.load('models/second_model_binary.pkl'),
            'shap1': joblib.load('models/shap_explainer_3class.pkl'),
            'shap2': joblib.load('models/shap_explainer_binary.pkl'),
            'demo_first': joblib.load('models/demo_dataset.pkl'),
            'demo': joblib.load('models/demo_bundle.pkl'),
        }
        with open('models/metrics.json', 'r') as f:
            data['metrics'] = json.load(f)
        return data
    except Exception as e:
        st.error(f"⚠️ Ошибка загрузки компонентов: {e}")
        return None

assets = load_assets()


# НАЧАЛЬНАЯ СТРАНИЦА
st.sidebar.image("https://img.icons8.com/?size=100&id=HFxwl7VaWPdE&format=png&color=000000", width=80)
st.sidebar.title("Oil & Gas")
page = st.sidebar.selectbox("Раздел:", [
    "🏠 Обзор системы", 
    "📊 Анализ данных (EDA)", 
    "🎯 Кандидаты (Бинарная модель)", 
    "🔍 Интерпретация SHAP"
])

if not assets:
    st.stop()

# ГЛАВНАЯ СТРАНИЦА
if page == "🏠 Обзор системы":
    st.title("🛢️ Двухступенчатая классификация скважин")

    # Функция, которая будет вызываться при нажатии на кнопку
    def set_random_idx():
        st.session_state.selected_idx = np.random.randint(0, len(assets['demo_first']['X_demo']))

    # Метрики
    col1, col2, col3 = st.columns(3)
    m1, m2 = assets['metrics']['model_3class'], assets['metrics']['model_binary']
    col1.metric("Точность (Stage 1)", f"{m1['accuracy']:.1%}")
    col2.metric("Точность (Stage 2)", f"{m2['accuracy']:.1%}")
    col3.metric("Скважин в демо-анализе", len(assets['demo_first']['X_demo']))

    st.divider()

    if 'selected_idx' not in st.session_state:
        st.session_state.selected_idx = 0
    
    st.subheader("🤖 Быстрый просмотр прогноза по скважине")
    sel_col, res_col = st.columns([1, 2])
    
    with sel_col:
        st.write("🔍 **Выбор объекта:**")
        # Ввод числа привязан к session_state
        idx = st.number_input("Введите ID (0-99):", 
                              min_value=0, 
                              max_value=len(assets['demo_first']['X_demo'])-1, 
                              key="selected_idx") # !
        
        # Кнопка просто вызывает функцию
        st.button("🎲 Случайный объект", on_click=set_random_idx)

    with res_col:
        res = assets['demo_first']['predictions_demo'][idx]
        colors = {-1: "#e74c3c", 0: "#f39c12", 1: "#27ae60"}
        labels = {-1: "НЕЭФФЕКТИВНАЯ", 0: "СПОРНАЯ (Нужен доп. анализ)", 1: "ЭФФЕКТИВНАЯ"}
        
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; border: 2px solid {colors[int(res)]}; background-color: white;">
                <h4 style="margin:0; color: #7f8c8d;">Результат модели для ID {idx}:</h4>
                <h2 style="color: {colors[int(res)]}; margin:0;">{labels[int(res)]}</h2>
            </div>
        """, unsafe_allow_html=True)

    # Матрица ошибок
    st.divider()
    st.subheader("📊 Качество классификации (Confusion Matrix)")
    
    y_true = assets['demo_first']['y_demo']
    y_pred = assets['demo_first']['predictions_demo']
    
    cm_col, text_col = st.columns([1, 1])
    
    with cm_col:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(cm, display_labels=['Неэфф.', 'Спорный', 'Эфф.']).plot(cmap='Blues', ax=ax)
        st.pyplot(fig)
        
    with text_col:
        st.write("### Что это значит?")
        st.write("""
        Матрица показывает, насколько часто модель ошибается в каждом из классов.
        - **Диагональ**: правильные ответы.
        - **Ошибки в 'Спорных'**: допустимы, так как они уходят на вторую ступень.
        - **Критическая ошибка**: если эффективная скважина(1) помечена как неэффективная(-1).
        """)

#  EDA
elif page == "📊 Анализ данных (EDA)":
    st.header("📊 Анализ признаков")
    X = assets['demo_first']['X_demo']
    
    tab1, tab2 = st.tabs(["Распределения", "Корреляции"])
    
    with tab1:
       feat = st.selectbox("Выберите параметр:", X.columns)
       fig = px.histogram(X, x=feat, color_discrete_sequence=['#00CC96'], marginal="box")
       st.plotly_chart(fig, use_container_width=True)
    with tab2:
        corr = X.select_dtypes('number').corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)


# КАНДИДАТЫ
elif page == "🎯 Кандидаты (Бинарная модель)":
    st.header("🎯 Итоговый отбор кандидатов")
    df_cand = pd.DataFrame(assets['demo']['candidates'])
    
    conf_threshold = st.sidebar.slider("Порог уверенности:", 0.5, 1.0, 0.85)
    
    final_list = df_cand[
        (df_cand['effect_pred'] == 1) & 
        (df_cand['confidence'] >= conf_threshold)
    ].sort_values('confidence', ascending=False)
    
    st.metric("Найдено топ-кандидатов", len(final_list))
    
    # Создаем два столбца: слева таблица, справа график
    tab_col, plot_col = st.columns([3, 2])
    
    with tab_col:
        st.write("📋 **Список приоритетных скважин**")
        st.dataframe(
            final_list[['well_num', 'inj_well_id', 'effect_pred_proba', 'confidence']], 
            use_container_width=True, 
            hide_index=True,
            height=400 # Фиксируем высоту, чтобы была прокрутка
        )
    
    with plot_col:
        if not final_list.empty:
            st.write("📈 **Анализ распределения уверенности**")
            fig = px.scatter(
                final_list, 
                x='effect_pred_proba', 
                y='confidence',
                hover_name='well_num',
                color='confidence',
                size='confidence',
                color_continuous_scale='Greens',
                labels={'effect_pred_proba': 'Вероятность успеха', 'confidence': 'Уверенность модели'}
            )
            # Убираем лишние отступы
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Нет данных для отображения. Снизьте порог уверенности.")

#  SHAP
elif page == "🔍 Интерпретация SHAP":
    st.header("🔍 Почему модель приняла решение?")
    
    target_model = st.radio("Выберите модель для анализа:", 
                            ["Первая ступень (3 класса)", "Вторая ступень (Бинарная)"], 
                            horizontal=True)
    
    if "Первая ступень" in target_model:
        exp = assets['shap1']['explainer']
        data_for_shap = assets['demo']['m1_data']
        class_idx = 2 # Анализируем класс "Эффективен" (индекс 2 в CatBoost)
    else:
        exp = assets['shap2']['explainer']
        data_for_shap = assets['demo']['m2_data']
        class_idx = None # Для бинарной модели индекс не нужен

    st.write(f"📊 Анализ на основе {data_for_shap.shape[1]} признаков")

    try:
        # Вычисляем SHAP values (Explanation object)
        shap_values = exp(data_for_shap)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Если это 3-классовая модель (3D массив), берем срез по классу
        if len(shap_values.shape) == 3 and class_idx is not None:
            shap.plots.beeswarm(shap_values[:, :, class_idx], max_display=15, show=False)
        else:
            shap.plots.beeswarm(shap_values, max_display=15, show=False)
        
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Ошибка SHAP: {e}")
        st.warning("Убедитесь, что данные соответствуют версиям моделей.")

st.sidebar.markdown("---")
st.sidebar.caption("Diplom Project © 2024")