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

# Configuração da simulação Monte Carlo com 500 rodadas
num_rodadas = 500
rss_values = []

for _ in range(num_rodadas):
    # Divisão aleatória dos dados em 80% para treino e 20% para teste
    n = len(potencia_gerada)
    indices = np.random.permutation(n)
    train_size = int(0.8 * n)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    y_train = potencia_gerada[train_indices]
    y_test = potencia_gerada[test_indices]
    
    # Calcular a média da potência gerada no conjunto de treino
    y_mean = np.mean(y_train)
    
    # Prever valores no conjunto de teste (todas as previsões são a média)
    y_pred = np.full(y_test.shape, y_mean)
    
    # Calcular RSS
    rss = np.sum((y_test - y_pred) ** 2)
    rss_values.append(rss)

# Calcular média, desvio-padrão, máximo e mínimo dos valores de RSS
rss_mean = np.mean(rss_values)
rss_std = np.std(rss_values)
rss_max = np.max(rss_values)
rss_min = np.min(rss_values)

print("Resultados da Simulação Monte Carlo para o modelo de Média de Valores Observáveis:")
print("Média do RSS:", rss_mean)
print("Desvio-padrão do RSS:", rss_std)
print("Maior valor de RSS:", rss_max)
print("Menor valor de RSS:", rss_min)
