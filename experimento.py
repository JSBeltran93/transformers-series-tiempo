import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

OUT = Path("resultados")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Serie de tiempo sintética: seno + tendencia + ruido
# ---------------------------------------------------------------------------
N = 1000
t = np.arange(N)
serie = (
    np.sin(0.05 * t) * 10
    + np.sin(0.011 * t) * 5
    + 0.01 * t
    + np.random.normal(0, 1.0, N)
)

plt.figure(figsize=(10, 3.5))
plt.plot(t, serie, linewidth=0.9, color="#1f77b4")
plt.title("Serie de tiempo sintética (seno + tendencia + ruido)")
plt.xlabel("Paso de tiempo")
plt.ylabel("Valor")
plt.tight_layout()
plt.savefig(OUT / "serie_original.png", dpi=130)
plt.close()

# Normalización
mu, sigma = serie.mean(), serie.std()
serie_norm = (serie - mu) / sigma

WINDOW = 30
HORIZON = 1


def construir_ventanas(serie, window, horizon):
    X, y = [], []
    for i in range(len(serie) - window - horizon + 1):
        X.append(serie[i : i + window])
        y.append(serie[i + window : i + window + horizon])
    return np.array(X), np.array(y)


X, y = construir_ventanas(serie_norm, WINDOW, HORIZON)
n_train = int(len(X) * 0.8)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # (N, W, 1)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

print(f"Ventanas de entrenamiento: {X_train_t.shape}, prueba: {X_test_t.shape}")

# ---------------------------------------------------------------------------
# 2. Mecanismo de atención (self-attention) implementado desde cero
# ---------------------------------------------------------------------------
class SelfAttention(nn.Module):
    """Self-attention de una sola cabeza, implementada explícitamente
    siguiendo Q, K, V y softmax(QK^T / sqrt(d_k)) V."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
        pesos_atencion = torch.softmax(scores, dim=-1)
        salida = torch.matmul(pesos_atencion, V)
        return salida, pesos_atencion


class ModeloAtencion(nn.Module):
    """Modelo simple: embedding lineal + self-attention + cabeza de predicción.
    Sirve para analizar directamente los pesos de atención sobre la serie."""

    def __init__(self, d_model=16, window=WINDOW):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(window, d_model) * 0.01)
        self.attn = SelfAttention(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 1))

    def forward(self, x):
        h = self.embed(x) + self.pos_embed
        salida, pesos = self.attn(h)
        ultimo = salida[:, -1, :]
        pred = self.head(ultimo)
        return pred, pesos


modelo_attn = ModeloAtencion()
opt_attn = torch.optim.Adam(modelo_attn.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

EPOCHS_ATTN = 150
historial_attn = []
for epoch in range(EPOCHS_ATTN):
    modelo_attn.train()
    opt_attn.zero_grad()
    pred, _ = modelo_attn(X_train_t)
    loss = loss_fn(pred, y_train_t)
    loss.backward()
    opt_attn.step()
    historial_attn.append(loss.item())
    if (epoch + 1) % 30 == 0:
        print(f"[Atención] Epoch {epoch+1}/{EPOCHS_ATTN} - loss_train: {loss.item():.5f}")

modelo_attn.eval()
with torch.no_grad():
    pred_test_attn, pesos_attn = modelo_attn(X_test_t)
    test_loss_attn = loss_fn(pred_test_attn, y_test_t).item()
print(f"[Atención] MSE en test: {test_loss_attn:.5f}")

# Graficar curva de pérdida
plt.figure(figsize=(7, 4))
plt.plot(historial_attn)
plt.title("Curva de entrenamiento — mecanismo de atención")
plt.xlabel("Época")
plt.ylabel("MSE (train)")
plt.tight_layout()
plt.savefig(OUT / "attn_loss.png", dpi=130)
plt.close()

# Visualizar mapa de atención de una ventana de ejemplo
idx_ejemplo = 5
mapa = pesos_attn[idx_ejemplo].numpy()
plt.figure(figsize=(6, 5))
plt.imshow(mapa, cmap="viridis", aspect="auto")
plt.colorbar(label="Peso de atención")
plt.title(f"Mapa de self-attention (ventana de prueba #{idx_ejemplo})")
plt.xlabel("Posición (key)")
plt.ylabel("Posición (query)")
plt.tight_layout()
plt.savefig(OUT / "attn_heatmap.png", dpi=130)
plt.close()

# Peso de atención que recibe cada paso pasado al predecir el siguiente valor (última fila)
atencion_sobre_pasado = mapa[-1]
plt.figure(figsize=(8, 3.5))
plt.bar(range(WINDOW), atencion_sobre_pasado, color="#ff7f0e")
plt.title("Atención que recibe cada paso pasado al predecir el siguiente valor")
plt.xlabel("Paso dentro de la ventana (0 = más antiguo, 29 = más reciente)")
plt.ylabel("Peso de atención")
plt.tight_layout()
plt.savefig(OUT / "attn_sobre_pasado.png", dpi=130)
plt.close()

pred_vs_real_attn = np.column_stack([
    y_test[:50].flatten(),
    pred_test_attn[:50].numpy().flatten(),
])
plt.figure(figsize=(10, 3.5))
plt.plot(pred_vs_real_attn[:, 0], label="Real", linewidth=1.2)
plt.plot(pred_vs_real_attn[:, 1], label="Predicho (atención)", linewidth=1.2, linestyle="--")
plt.title("Predicción vs. valor real (modelo de atención) — primeras 50 muestras de test")
plt.xlabel("Muestra de test")
plt.ylabel("Valor normalizado")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "attn_pred_vs_real.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------------
# 3. Transformer básico (encoder) para la misma tarea de predicción
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerSerieTiempo(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, window=WINDOW):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=window)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        h = self.embed(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        ultimo = h[:, -1, :]
        return self.head(ultimo)


modelo_tf = TransformerSerieTiempo()
opt_tf = torch.optim.Adam(modelo_tf.parameters(), lr=1e-3)

EPOCHS_TF = 200
historial_tf = []
for epoch in range(EPOCHS_TF):
    modelo_tf.train()
    opt_tf.zero_grad()
    pred = modelo_tf(X_train_t)
    loss = loss_fn(pred, y_train_t)
    loss.backward()
    opt_tf.step()
    historial_tf.append(loss.item())
    if (epoch + 1) % 40 == 0:
        print(f"[Transformer] Epoch {epoch+1}/{EPOCHS_TF} - loss_train: {loss.item():.5f}")

modelo_tf.eval()
with torch.no_grad():
    pred_test_tf = modelo_tf(X_test_t)
    test_loss_tf = loss_fn(pred_test_tf, y_test_t).item()
print(f"[Transformer] MSE en test: {test_loss_tf:.5f}")

plt.figure(figsize=(7, 4))
plt.plot(historial_tf, color="#2ca02c")
plt.title("Curva de entrenamiento — Transformer básico")
plt.xlabel("Época")
plt.ylabel("MSE (train)")
plt.tight_layout()
plt.savefig(OUT / "tf_loss.png", dpi=130)
plt.close()

plt.figure(figsize=(10, 3.5))
plt.plot(y_test[:50].flatten(), label="Real", linewidth=1.2)
plt.plot(pred_test_tf[:50].numpy().flatten(), label="Predicho (Transformer)", linewidth=1.2, linestyle="--")
plt.title("Predicción vs. valor real (Transformer) — primeras 50 muestras de test")
plt.xlabel("Muestra de test")
plt.ylabel("Valor normalizado")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "tf_pred_vs_real.png", dpi=130)
plt.close()

# Comparación final de ambos modelos
plt.figure(figsize=(10, 3.5))
plt.plot(y_test[:50].flatten(), label="Real", linewidth=1.4, color="black")
plt.plot(pred_test_attn[:50].numpy().flatten(), label="Atención simple", linestyle="--")
plt.plot(pred_test_tf[:50].numpy().flatten(), label="Transformer", linestyle=":")
plt.title("Comparación: atención simple vs. Transformer — primeras 50 muestras de test")
plt.xlabel("Muestra de test")
plt.ylabel("Valor normalizado")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "comparacion_modelos.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------------
# Resumen numérico para el informe (nunca inventado, siempre leído de aquí)
# ---------------------------------------------------------------------------
resumen = {
    "n_muestras_serie": N,
    "ventana": WINDOW,
    "n_train": len(X_train),
    "n_test": len(X_test),
    "epochs_atencion": EPOCHS_ATTN,
    "mse_train_final_atencion": historial_attn[-1],
    "mse_test_atencion": test_loss_attn,
    "epochs_transformer": EPOCHS_TF,
    "mse_train_final_transformer": historial_tf[-1],
    "mse_test_transformer": test_loss_tf,
}
with open(OUT / "resumen.txt", "w") as f:
    for k, v in resumen.items():
        f.write(f"{k}: {v}\n")

print("\nResumen final:")
for k, v in resumen.items():
    print(f"  {k}: {v}")
