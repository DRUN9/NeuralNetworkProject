from NeuralNetwork import NeuralNetwork, inverse_network, test_network, train_network, INPUT_NODES, OUTPUT_NODES
import os.path


def experiment():
    print("Выберите критерий оценивания работы сети")
    print("0 - по скрытым узлам, 1 - по количеству эпох, 2 - по коэффициенту обучения")
    criterion = int(input())
    if criterion == 0:
        print("Введите коэффициент обучения")
        learning_rate = int(input())
        print("Введите количество эпох")
        epochs = int(input())
        for hidden_nodes in range(50, 550, 50):
            network = NeuralNetwork(INPUT_NODES, hidden_nodes, OUTPUT_NODES, learning_rate)
            train_network(network, epochs)
            effectiveness = test_network(network)
            with open(os.path.join("Результаты тестов", "По скрытым узлам", "Результаты.txt"), 'a') as file:
                file.write(str(hidden_nodes) + ' ' + str(effectiveness) + '\n')
            for i in range(10):
                inverse_network(network, i, hidden_nodes, criterion, exp=True)
    elif criterion == 1:
        print("Введите количество скрытых узлов")
        hidden_nodes = int(input())
        print("Введите коэффициент обучения")
        learning_rate = int(input())
        for epochs in range(1, 10):
            network = NeuralNetwork(INPUT_NODES, hidden_nodes, OUTPUT_NODES, learning_rate)
            train_network(network, epochs)
            effectiveness = test_network(network)
            with open(os.path.join("Результаты тестов", "По количеству эпох", "Результаты.txt"), 'a') as file:
                file.write(str(epochs) + ' ' + str(effectiveness) + '\n')
            for i in range(10):
                inverse_network(network, i, epochs, criterion, exp=True)
    elif criterion == 2:
        print("Введите количество скрытых узлов")
        hidden_nodes = int(input())
        print("Введите количество эпох")
        epochs = int(input())
        for i in range(1, 10):
            learning_rate = i / 10
            network = NeuralNetwork(INPUT_NODES, hidden_nodes, OUTPUT_NODES, learning_rate)
            train_network(network, epochs)
            effectiveness = test_network(network)
            with open(os.path.join("Результаты тестов", "По коэффициенту обучения", "Результаты.txt"), 'a') as file:
                file.write(str(learning_rate) + ' ' + str(effectiveness) + '\n')
            for j in range(10):
                inverse_network(network, j, learning_rate, criterion, exp=True)
