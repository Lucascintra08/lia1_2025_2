import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

# =========================
# CONFIGURAÇÃO
# =========================
st.set_page_config(page_title="Energia Residencial – Regressão & Classificação", layout="wide")
st.title("Previsão do consumo de Energia Residencial:")

# Caminhos dos arquivos (ajuste se quiser usar locais diferentes)
TRAIN_PATH = r"C:\Users\lucas\OneDrive\Anexos\Documentos\LIA1\streamlit\Projeto_Streamlit\train_energy_data.csv"
TEST_PATH  = r"C:\Users\lucas\OneDrive\Anexos\Documentos\LIA1\streamlit\Projeto_Streamlit\test_energy_data.csv"

# =========================
# 1) CARREGAR DADOS
# =========================
@st.cache_data
def load_csv_auto(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path)

try:
    train = load_csv_auto(TRAIN_PATH)
    test  = load_csv_auto(TEST_PATH)
except Exception as e:
    st.error(f"Erro ao ler os CSVs: {e}")
    st.stop()

if "Square Footage" not in train.columns or "Square Footage" not in test.columns:
    st.error("Coluna 'Square Footage' não encontrada nos arquivos.")
    st.stop()

# Conversão de pés² para m² com 2 casas decimais
train["Square Footage"] = (train["Square Footage"] * 0.092903)
test["Square Footage"]  = (test["Square Footage"]  * 0.092903)


# Renomear colunas para português
colunas_novas = {
    "Building Type": "Tipo de Edificação",
    "Square Footage": "Área Construída (m²)",
    "Number of Occupants": "Nº de Ocupantes",
    "Appliances Used": "Aparelhos Utilizados",
    "Average Temperature": "Temperatura Média(°C)",
    "Day of Week": "Dia da Semana",
    "Energy Consumption": "Consumo de Energia (kWh)"
}
train = train.rename(columns=colunas_novas)
test  = test.rename(columns=colunas_novas)

target_col = "Consumo de Energia (kWh)"
if target_col not in train.columns:
    st.error(f"Coluna alvo '{target_col}' não encontrada após renomear.")
    st.stop()

num_cols = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c]) and c != target_col]
cat_cols = [c for c in train.columns if (train[c].dtype == "object" or str(train[c].dtype).startswith("category")) and c != target_col]

if not num_cols:
    st.error("Não há colunas numéricas preditoras além do alvo.")
    st.stop()

st.subheader("Amostra do dataset")
st.table(train.head())

# =========================
# 2) REGRESSÃO LINEAR 1D
# =========================
st.header("📈 Regressão Linear:")

# Escolha da variável numérica 
x_col = st.selectbox("Variável numérica para a regressão:", num_cols, index=0)

# Treina no TREINO e avalia no TESTE
X_train = train[[x_col]].dropna().values
y_train = train.loc[train[[x_col]].dropna().index, target_col].astype(float).values

X_test  = test[[x_col]].dropna().values
y_test  = test.loc[test[[x_col]].dropna().index, target_col].astype(float).values

reg_model = LinearRegression().fit(X_train, y_train)
y_pred_test = reg_model.predict(X_test)

# Layout 2 colunas (dados + gráfico)
col1, col2 = st.columns(2)
with col1:
    st.subheader("Dados (treino)")
    st.table(train[[x_col, target_col]].head())

with col2:
    st.subheader("Gráfico de Dispersão + Reta (treino)")
    Xp = train[[x_col]].values
    yp = train[target_col].values
    x_grid = np.linspace(np.nanmin(Xp), np.nanmax(Xp), 200).reshape(-1, 1)
    y_hat  = reg_model.predict(x_grid)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(Xp, yp, color="blue", s=10)
    ax.plot(x_grid, y_hat, color="red")
    ax.set_xlabel(x_col)
    ax.set_ylabel(target_col)
    st.pyplot(fig)

# Predição manual (number_input + botão)
st.subheader(f"Prever {target_col} a partir de {x_col}")
xmin = float(np.nanpercentile(train[x_col], 1))
xmax = float(np.nanpercentile(train[x_col], 99))
xmed = float(np.nanmedian(train[x_col]))
new_x = st.number_input(f"Insira {x_col}", min_value=xmin, max_value=xmax, value=xmed, step=1.0)
if st.button("Processar Regressão"):
    pred = reg_model.predict(np.array([[new_x]]))[0]
    st.header(f"Previsão de {target_col}: {pred:.2f}")

st.markdown("---")

# =========================
# 3) CLASSIFICAÇÃO (Baixo / Médio / Alto)
# =========================
st.header("Classificação do Consumo (Baixo / Médio / Alto)")

# Cria classes por tercis no TREINO e aplica no TESTE
q = train[target_col].quantile([1/3, 2/3]).values
q1, q2 = q[0], q[1]

def to_class(v):
    if v <= q1: return "Baixo"
    if v <= q2: return "Médio"
    return "Alto"

train = train.copy()
test  = test.copy()
train["consumption_class"] = train[target_col].apply(to_class)
test["consumption_class"]  = test[target_col].apply(to_class)

# Para manter o estilo do seu 2º código (NB categórico), criamos BINS p/ numéricos
st.sidebar.header("Classificação – configurações")
bins_slider = st.sidebar.slider("Nº de faixas (bins) para numéricos", 3, 8, 4)

binned_cols = []
for c in num_cols:
    try:
        # Gera os bins
        cats, bins = pd.qcut(train[c], q=bins_slider, retbins=True, duplicates="drop")
        
        # Arredonda os limites dos bins
        bins = np.round(bins, 2)
        
        # Cria rótulos personalizados com duas casas decimais
        labels = [f"{bins[i]} a {bins[i+1]}." for i in range(len(bins)-1)]
        
        # Aplica os bins com labels formatados
        train[f"{c}"] = pd.cut(train[c], bins=bins, include_lowest=True, labels=labels)
        test[f"{c}"]  = pd.cut(test[c],  bins=bins, include_lowest=True, labels=labels)
        
        binned_cols.append(f"{c}")
    except Exception:
        pass

# Features de classificação = categóricas originais + numéricas "binned"
clf_features = cat_cols + binned_cols
if not clf_features:
    st.warning("Não há features categóricas (nem bins) disponíveis para a classificação.")
else:
    Xtr = train[clf_features].astype(str)
    ytr = train["consumption_class"].astype("category").cat.codes

    Xte = test[clf_features].astype(str)
    yte = test["consumption_class"].astype("category").cat.codes

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xtr_enc = encoder.fit_transform(Xtr)
    Xte_enc = encoder.transform(Xte)

    clf_model = CategoricalNB().fit(Xtr_enc, ytr)
    y_pred = clf_model.predict(Xte_enc)
    acc = accuracy_score(yte, y_pred) if len(yte) else np.nan
    st.write(f"**Acurácia (teste):** {acc:.2f}")

    # UI de entrada (selectbox + botão)
    inputs = []
    cols_form = st.columns(3)
    for i, c in enumerate(clf_features):
        with cols_form[i % 3]:
            options = sorted(train[c].dropna().astype(str).unique().tolist())
            val = st.selectbox(c, options)
            inputs.append(val)

    if st.button("Processar Classificação"):
        X_in = pd.DataFrame([inputs], columns=clf_features)
        X_in_enc = encoder.transform(X_in)
        pred_code = clf_model.predict(X_in_enc)[0]
        classes = train["consumption_class"].astype("category").cat.categories
        label = classes[pred_code]
        st.header(f"Resultado: {label}")
