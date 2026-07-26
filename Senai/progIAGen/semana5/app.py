"""
Dashboard de Manutenção Preditiva Industrial
=============================================
Consolida as Etapas 1, 2 e 3 do projeto:
  - Etapa 1: EDA, tratamento de dados, Decision Tree
  - Etapa 2: Feature Engineering, Cross-Validation, Grid Search, Random Forest
  - Etapa 3: Diagnóstico automático com IA Generativa (Gemini) - opcional

Para rodar:
    pip install -r requirements.txt
    streamlit run dashboard_manutencao_preditiva.py
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Manutenção Preditiva Industrial",
    page_icon="🏭",
    layout="wide",
)

COLS = ["temperatura", "vibracao", "pressao", "tempo_operacao"]


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
@st.cache_data
def carregar_dados_exemplo():
    df=pd.read_csv("dataset_sensores_industriais_200_registros.csv")
    return df

@st.cache_data
def tratar_dados(df):
    df_tratado = df.copy()
    df_tratado = df_tratado.fillna(df_tratado.mean(numeric_only=True))
    return df_tratado


@st.cache_data
def criar_features_avancadas(df_tratado):
    X = df_tratado.drop("falha", axis=1).copy()
    y = df_tratado["falha"]
    X["termica_pressao"] = X["temperatura"] * X["pressao"]
    X["desgaste_acumulado"] = X["vibracao"] * X["tempo_operacao"]
    return X, y


@st.cache_resource
def treinar_modelos(X, y, usar_grid_search=True):
    resultados = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Decision Tree simples (Etapa 1)
    dt = DecisionTreeClassifier(random_state=42, max_depth=4)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    resultados["decision_tree"] = {
        "modelo": dt,
        "acuracia": accuracy_score(y_test, y_pred_dt),
        "matriz": confusion_matrix(y_test, y_pred_dt),
        "relatorio": classification_report(y_test, y_pred_dt, zero_division=0),
    }

    # Cross-validation (Etapa 2)
    scores = cross_val_score(
        DecisionTreeClassifier(random_state=42), X, y, cv=5, scoring="accuracy"
    )
    resultados["cv_scores"] = scores

    # Grid Search (Etapa 2)
    if usar_grid_search:
        param_grid = {
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"],
        }
        grid = GridSearchCV(
            DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring="accuracy"
        )
        grid.fit(X, y)
        resultados["grid_search"] = grid

    # Random Forest (Etapa 2)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    resultados["random_forest"] = {
        "modelo": rf,
        "acuracia": accuracy_score(y_test, y_pred_rf),
        "matriz": confusion_matrix(y_test, y_pred_rf),
        "relatorio": classification_report(y_test, y_pred_rf, zero_division=0),
    }

    resultados["X_test"] = X_test
    resultados["y_test"] = y_test
    return resultados


def gerar_prompt_diagnostico(registro):
    return f"""
Você é um especialista em manutenção industrial.

Analise os dados abaixo:

Temperatura: {registro['temperatura']:.2f}
Vibração: {registro['vibracao']:.2f}
Pressão: {registro['pressao']:.2f}
Tempo de operação: {registro['tempo_operacao']:.2f}

Explique:
1. Se existe risco de falha;
2. Qual sensor aparenta maior criticidade;
3. Sugestões de manutenção preventiva.
"""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🏭 Manutenção Preditiva")
st.sidebar.markdown("Painel unificado das Etapas 1, 2 e 3 do projeto.")

arquivo = st.sidebar.file_uploader("Envie o CSV de sensores", type="csv")
usar_exemplo = st.sidebar.checkbox("Usar dados de exemplo (sintéticos)", value=arquivo is None)

if arquivo is not None:
    df_raw = pd.read_csv(arquivo)
elif usar_exemplo:
    df_raw = carregar_dados_exemplo()
else:
    st.warning("Envie um CSV ou marque a opção de dados de exemplo na barra lateral.")
    st.stop()

colunas_esperadas = COLS + ["falha"]
faltando = [c for c in colunas_esperadas if c not in df_raw.columns]
if faltando:
    st.error(f"O arquivo enviado não contém as colunas esperadas: {faltando}")
    st.stop()

pagina = st.sidebar.radio(
    "Navegação",
    [
        "📊 Visão Geral & EDA",
        "🧹 Tratamento & Outliers",
        "🧠 Modelos (Etapa 1 e 2)",
        "🔍 Predição Manual",
        "🤖 Diagnóstico com IA (Gemini)",
    ],
)

df_tratado = tratar_dados(df_raw)
X_avancado, y = criar_features_avancadas(df_tratado)

# ---------------------------------------------------------------------------
# Página 1: Visão Geral & EDA
# ---------------------------------------------------------------------------
if pagina == "📊 Visão Geral & EDA":
    st.title("📊 Visão Geral do Dataset")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", df_raw.shape[0])
    c2.metric("Colunas", df_raw.shape[1])
    c3.metric("Taxa de falha", f"{df_tratado['falha'].mean():.1%}")
    c4.metric("Valores faltantes", int(df_raw.isnull().sum().sum()))

    st.subheader("Amostra dos dados")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.subheader("Estatísticas descritivas")
    st.dataframe(df_raw.describe(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Valores faltantes por coluna")
        st.bar_chart(df_raw.isnull().sum())
    with col2:
        st.subheader("Distribuição da variável alvo (falha)")
        st.bar_chart(df_tratado["falha"].value_counts())

    st.subheader("Distribuição das variáveis numéricas")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, col in zip(axes, COLS):
        ax.hist(df_tratado[col], bins=20, color="#3b6ea5")
        ax.set_title(col)
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# Página 2: Tratamento & Outliers
# ---------------------------------------------------------------------------
elif pagina == "🧹 Tratamento & Outliers":
    st.title("🧹 Tratamento de Dados Faltantes e Outliers")

    st.markdown(
        "Estratégia aplicada: preenchimento de valores faltantes com a "
        "**média da coluna** (abordagem didática da Etapa 1)."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Antes do tratamento")
        st.dataframe(df_raw.isnull().sum().rename("faltantes"))
    with col2:
        st.subheader("Depois do tratamento")
        st.dataframe(df_tratado.isnull().sum().rename("faltantes"))

    st.subheader("Boxplot — observação de outliers")
    fig, ax = plt.subplots(figsize=(10, 5))
    df_tratado[COLS].boxplot(ax=ax)
    ax.set_ylabel("Valores")
    st.pyplot(fig)

    st.subheader("Engenharia de Recursos (Etapa 2)")
    st.markdown(
        "- `termica_pressao` = temperatura × pressão (estresse físico-térmico)\n"
        "- `desgaste_acumulado` = vibração × tempo de operação (desgaste ao longo da vida útil)"
    )
    st.dataframe(X_avancado.head(10), use_container_width=True)

# ---------------------------------------------------------------------------
# Página 3: Modelos
# ---------------------------------------------------------------------------
elif pagina == "🧠 Modelos (Etapa 1 e 2)":
    st.title("🧠 Treinamento e Avaliação dos Modelos")

    usar_grid = st.checkbox("Executar Grid Search (pode levar alguns segundos)", value=True)
    with st.spinner("Treinando modelos..."):
        resultados = treinar_modelos(X_avancado, y, usar_grid_search=usar_grid)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Decision Tree", "Cross-Validation", "Random Forest", "Importância das Variáveis"]
    )

    with tab1:
        st.subheader("Árvore de Decisão (max_depth=4)")
        st.metric("Acurácia", f"{resultados['decision_tree']['acuracia']:.2%}")
        st.text("Matriz de confusão")
        st.dataframe(pd.DataFrame(resultados["decision_tree"]["matriz"]))
        st.text("Relatório de classificação")
        st.code(resultados["decision_tree"]["relatorio"])

    with tab2:
        st.subheader("Validação Cruzada (K-Fold, 5 dobras)")
        scores = resultados["cv_scores"]
        st.write("Acurácias por fold:", scores)
        c1, c2 = st.columns(2)
        c1.metric("Acurácia média", f"{scores.mean():.2%}")
        c2.metric("Desvio padrão", f"{scores.std():.4f}")

        if "grid_search" in resultados:
            st.subheader("Grid Search")
            st.json(resultados["grid_search"].best_params_)
            st.metric(
                "Melhor acurácia (CV)",
                f"{resultados['grid_search'].best_score_:.2%}",
            )

    with tab3:
        st.subheader("Random Forest (100 árvores)")
        st.metric("Acurácia", f"{resultados['random_forest']['acuracia']:.2%}")
        st.text("Matriz de confusão")
        st.dataframe(pd.DataFrame(resultados["random_forest"]["matriz"]))
        st.text("Relatório de classificação")
        st.code(resultados["random_forest"]["relatorio"])

    with tab4:
        st.subheader("Importância dos Sensores")
        rf = resultados["random_forest"]["modelo"]
        df_imp = pd.DataFrame({
            "Sensor": X_avancado.columns,
            "Importância": rf.feature_importances_,
        }).sort_values("Importância", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(df_imp["Sensor"], df_imp["Importância"], color="#3b6ea5")
        ax.set_xlabel("Grau de importância (0 a 1)")
        st.pyplot(fig)

    st.session_state["resultados_modelos"] = resultados

# ---------------------------------------------------------------------------
# Página 4: Predição manual
# ---------------------------------------------------------------------------
elif pagina == "🔍 Predição Manual":
    st.title("🔍 Simular uma Nova Leitura de Sensores")

    if "resultados_modelos" not in st.session_state:
        with st.spinner("Treinando modelos..."):
            st.session_state["resultados_modelos"] = treinar_modelos(
                X_avancado, y, usar_grid_search=False
            )
    resultados = st.session_state["resultados_modelos"]

    c1, c2 = st.columns(2)
    with c1:
        temperatura = st.slider("Temperatura", 40.0, 180.0, 82.0)
        vibracao = st.slider("Vibração", 0.0, 16.0, 4.6)
    with c2:
        pressao = st.slider("Pressão", 20.0, 46.0, 36.0)
        tempo_operacao = st.slider("Tempo de operação", 40.0, 260.0, 210.0)

    nova_amostra = pd.DataFrame([{
        "temperatura": temperatura,
        "vibracao": vibracao,
        "pressao": pressao,
        "tempo_operacao": tempo_operacao,
        "termica_pressao": temperatura * pressao,
        "desgaste_acumulado": vibracao * tempo_operacao,
    }])[X_avancado.columns]

    modelo_escolhido = st.selectbox("Modelo para predição", ["Random Forest", "Decision Tree"])
    modelo = (
        resultados["random_forest"]["modelo"]
        if modelo_escolhido == "Random Forest"
        else resultados["decision_tree"]["modelo"]
    )

    pred = modelo.predict(nova_amostra)[0]
    proba = modelo.predict_proba(nova_amostra)[0][1]

    if pred == 0:
        st.success(f"✅ Funcionamento normal (probabilidade de falha: {proba:.1%})")
    else:
        st.error(f"⚠️ Possível falha detectada (probabilidade de falha: {proba:.1%})")

    st.session_state["ultimo_registro"] = {
        "temperatura": temperatura,
        "vibracao": vibracao,
        "pressao": pressao,
        "tempo_operacao": tempo_operacao,
    }

# ---------------------------------------------------------------------------
# Página 5: IA Generativa (Etapa 3)
# ---------------------------------------------------------------------------
elif pagina == "🤖 Diagnóstico com IA (Gemini)":
    st.title("🤖 Diagnóstico Automático com IA Generativa (Gemini)")

    st.markdown(
        "Esta seção corresponde à **Etapa 3** do projeto. Informe sua chave de API "
        "do [Google AI Studio](https://aistudio.google.com/) para gerar diagnósticos "
        "automáticos em linguagem natural a partir dos dados dos sensores."
    )

    api_key = st.text_input("GOOGLE_API_KEY", type="password")
    indice = st.number_input(
        "Índice do registro a analisar", min_value=0, max_value=len(df_tratado) - 1, value=10
    )
    registro = df_tratado.iloc[indice]

    st.dataframe(registro.to_frame("valor"))

    if st.button("Gerar diagnóstico com IA"):
        if not api_key:
            st.warning("Informe sua chave de API para continuar.")
        else:
            try:
                import google.generativeai as genai

                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.0-flash")
                prompt = gerar_prompt_diagnostico(registro)
                with st.spinner("Consultando o Gemini..."):
                    response = model.generate_content(prompt)
                st.markdown(response.text)
            except ImportError:
                st.error(
                    "Biblioteca `google-generativeai` não instalada. "
                    "Rode: pip install google-generativeai"
                )
            except Exception as e:
                st.error(f"Erro ao chamar a API do Gemini: {e}")

    st.divider()
    st.subheader("Gerar relatório em lote e salvar em arquivo")
    n_registros = st.slider("Quantidade de registros para o relatório", 1, 20, 5)
    if st.button("Gerar relatório em lote"):
        if not api_key:
            st.warning("Informe sua chave de API para continuar.")
        else:
            try:
                import google.generativeai as genai

                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.0-flash")
                buffer = io.StringIO()
                progresso = st.progress(0)
                for i in range(n_registros):
                    reg = df_tratado.iloc[i]
                    prompt = gerar_prompt_diagnostico(reg)
                    response = model.generate_content(prompt)
                    buffer.write(f"===== REGISTRO {i} =====\n")
                    buffer.write(response.text)
                    buffer.write("\n\n")
                    progresso.progress((i + 1) / n_registros)
                st.success("Relatório gerado com sucesso!")
                st.download_button(
                    "📥 Baixar relatorio_diagnosticos.txt",
                    data=buffer.getvalue(),
                    file_name="relatorio_diagnosticos.txt",
                )
            except ImportError:
                st.error(
                    "Biblioteca `google-generativeai` não instalada. "
                    "Rode: pip install google-generativeai"
                )
            except Exception as e:
                st.error(f"Erro ao chamar a API do Gemini: {e}")
