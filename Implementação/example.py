import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Função para inicializar a população de pesos do AG
def initialize_population(population_size, num_weights):
    return np.random.rand(population_size, num_weights)

# Função de avaliação (fitness)
def evaluate_fitness(population, X_train, y_train):
    fitness_values = []
    for weights in population:
        # Substitua a implementação do Perceptron aqui
        predictions = perceptron(X_train, weights)
        fitness = accuracy_score(y_train, predictions)  # Pode ser substituído por outra métrica
        fitness_values.append(fitness)
    return np.array(fitness_values)

# Função de crossover (recombinação)
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# Função de mutação
def mutate(weights, mutation_rate):
    mutation_mask = np.random.rand(len(weights)) < mutation_rate
    weights[mutation_mask] += np.random.randn(np.sum(mutation_mask))  # Adicione mutação gaussiana
    return weights

# Função do Perceptron (substitua pela sua implementação)
def perceptron(X, weights):
    return np.dot(X, weights) > 0  # Exemplo simples; ajuste conforme necessário

# Função principal do AG
def genetic_algorithm(X_train, y_train, population_size, num_generations, mutation_rate):
    num_weights = X_train.shape[1]
    population = initialize_population(population_size, num_weights)

    for generation in range(num_generations):
        fitness_values = evaluate_fitness(population, X_train, y_train)

        # Seleção dos pais com base no fitness
        parents_indices = np.argsort(fitness_values)[-2:]
        parents = population[parents_indices]

        # Recombinação (crossover)
        children = []
        for _ in range(population_size // 2):
            child1, child2 = crossover(parents[0], parents[1])
            children.extend([child1, child2])
        
        children = np.array(children)

        # Mutação
        for i in range(population_size):
            children[i] = mutate(children[i], mutation_rate)

        # Substituição da população antiga pela nova (incluindo pais e filhos)
        population = np.vstack([parents, children])

    # Seleção do melhor indivíduo
    best_individual = population[np.argmax(evaluate_fitness(population, X_train, y_train))]

    return best_individual

# Exemplo de uso com a base de dados Iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Parâmetros do AG
population_size = 50
num_generations = 100
mutation_rate = 0.1

# Execução do AG
best_weights = genetic_algorithm(X_train, y_train, population_size, num_generations, mutation_rate)

# Avaliação do modelo final
predictions = perceptron(X_test, best_weights)
accuracy = accuracy_score(y_test, predictions)

print(f'Acurácia do modelo final: {accuracy}')
