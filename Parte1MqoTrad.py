import numpy as np
import matplotlib.pyplot as plt

# Carregar o arquivo de dados
data = np.loadtxt('aerogerador.dat')

# Separar as variáveis independentes (velocidade do vento) e dependentes (potência gerada)
velocidade_vento = data[:, 0]
potencia_gerada = data[:, 1]

# Visualizar os dados com um gráfico de dispersão
plt.figure(figsize=(8, 6))
plt.scatter(velocidade_vento, potencia_gerada, color='green', alpha=0.6, edgecolor='k')
plt.title('Gráfico de Dispersão: Velocidade do Vento vs Potência Gerada')
plt.xlabel('Velocidade do Vento')
plt.ylabel('Potência Gerada')
plt.grid(True)
plt.show()

# Organizar os dados na matriz X e no vetor y para regressão
n = len(velocidade_vento)
X = np.c_[np.ones(n), velocidade_vento]  # Matriz X com coluna de 1's para o intercepto
y = potencia_gerada.reshape(-1, 1)       # Vetor y com potência gerada

# Implementar o modelo de MQO tradicional para estimar os coeficientes
beta_mqo = np.linalg.inv(X.T @ X) @ X.T @ y
print("Coeficientes do modelo MQO tradicional:", beta_mqo.flatten())

# Configuração da simulação Monte Carlo com 500 rodadas
num_rodadas = 500
rss_values = []

for _ in range(num_rodadas):
    # Divisão aleatória dos dados em 80% para treino e 20% para teste
    indices = np.random.permutation(n)
    train_size = int(0.8 * n)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # Estimar os coeficientes no conjunto de treino
    beta_train = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    
    # Prever valores no conjunto de teste e calcular RSS
    y_pred = X_test @ beta_train
    rss = np.sum((y_test - y_pred) ** 2)
    rss_values.append(rss)

# Calcular média, desvio-padrão, máximo e mínimo dos valores de RSS
rss_mean = np.mean(rss_values)
rss_std = np.std(rss_values)
rss_max = np.max(rss_values)
rss_min = np.min(rss_values)

print("Resultados da Simulação Monte Carlo para o MQO tradicional:")
print("Média do RSS:", rss_mean)
print("Desvio-padrão do RSS:", rss_std)
print("Maior valor de RSS:", rss_max)
print("Menor valor de RSS:", rss_min)
