import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Olá, streamlit!")

st.write("Este é meu primeiro aplicativo usando Streamlit.")

st.header("Seção de Cabeçalho")

st.subheader("Subseção")

st.text("Este é um texto simples.")

st.markdown("Este é um texto **negrito** e *itálico* .")

data={
    'Nome': ['Ana', 'Bruno', 'Carlos'],
    'Salário': [5000, 6000, 7000],
    'Idade': [25, 30, 35],
    'Cidade': ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte']}





fig, ax = plt.subplots()
ax.bar(data['Nome'], data['Salário'])
st.pyplot(fig)



df = pd.DataFrame(data)
st.dataframe(df)


if st.button("Clique aqui"):
    st.write("Você clicou no botão!")



idade = st.slider("Selecione a idade", 0, 100)
st.write("Idade selecionada:", idade)

opcao = st.selectbox("Departamento", ["Vendas", "Marketing", "TI", "Recursos Humanos"])
st.write("Departamento selecionado:", opcao)



