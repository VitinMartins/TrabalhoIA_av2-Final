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

# Definindo os valores de lambda a serem testados, incluindo λ = 0
lambdas = [0, 0.25, 0.5, 0.75, 1]
num_rodadas = 500
rss_results = {lmbda: [] for lmbda in lambdas}

# Configuração da simulação Monte Carlo
n = len(potencia_gerada)

for _ in range(num_rodadas):
    # Divisão aleatória dos dados em 80% para treino e 20% para teste
    indices = np.random.permutation(n)
    train_size = int(0.8 * n)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train = np.c_[np.ones(train_size), velocidade_vento[train_indices]]  # Matriz X com coluna de 1's
    y_train = potencia_gerada[train_indices].reshape(-1, 1)                # Vetor y com potência gerada
    X_test = np.c_[np.ones(n - train_size), velocidade_vento[test_indices]] # Matriz X para teste
    y_test = potencia_gerada[test_indices].reshape(-1, 1)                   # Vetor y para teste
    
    # Testar cada valor de lambda
    for lmbda in lambdas:
        # Estimar os coeficientes com regularização
        beta_ridge = np.linalg.inv(X_train.T @ X_train + lmbda * np.identity(X_train.shape[1])) @ X_train.T @ y_train
        
        # Prever valores no conjunto de teste e calcular RSS
        y_pred = X_test @ beta_ridge
        rss = np.sum((y_test - y_pred) ** 2)
        rss_results[lmbda].append(rss)

# Calcular média, desvio-padrão, máximo e mínimo dos valores de RSS para cada lambda
for lmbda in lambdas:
    rss_mean = np.mean(rss_results[lmbda])
    rss_std = np.std(rss_results[lmbda])
    rss_max = np.max(rss_results[lmbda])
    rss_min = np.min(rss_results[lmbda])
    
    print(f"Resultados da Simulação Monte Carlo para λ = {lmbda}:")
    print(f"Média do RSS: {rss_mean:.4f}")
    print(f"Desvio-padrão do RSS: {rss_std:.4f}")
    print(f"Maior valor de RSS: {rss_max:.4f}")
    print(f"Menor valor de RSS: {rss_min:.4f}")
    print()

# Mostrar os coeficientes estimados para cada lambda
for lmbda in lambdas:
    print(f"Estimativas de β para λ = {lmbda}:")
    beta_ridge = np.linalg.inv(X_train.T @ X_train + lmbda * np.identity(X_train.shape[1])) @ X_train.T @ y_train
    print(beta_ridge.flatten())
    print()
