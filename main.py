import numpy
import matplotlib.pyplot
from NeuralNetwork import NeuralNetwork

# Работа с нейронной сетью на основе данных MNIST

# Тренировочный набор
# https://www.pjreddie.com/media/files/mnist_train.csv

# Тестовый набор
# https://www.pjreddie.com/media/files/mnist_test.csv

# 10 записей из тестового набора данных MNIST
# https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_test_10.csv

# 100 записей из тренировочного набора данных MNIST
# https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_train_100.csv

# Создать экземпляр нейронной сети
# Входных узлов - 784, скрытых - 100, выходных - 100
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.2
network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Загрузить в список тестовый набор данных CSV-файла набора MNIST
training_data_file = open("MNIST/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Тренировка нейронной сети

# Переменная epochs указывает, сколько раз тренировочный набор данных используется для тренировки сети
epochs = 2
for i in range(epochs):
    # Перебрать все записи в тренировочном наборе данных
    for record in training_data_list:
        # Получить список значений
        all_values_tr = record.split(',')
        # Масштабировать и сместить входные значения
        inputs = (numpy.asfarray(all_values_tr[1:]) / 255.0 * 0.99) + 0.01
        # Создать целевые выходные значения(все равны 0.01, за исключением желаемого маркерного значения, равного 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values_tr[0])] = 0.99
        network.train(inputs, targets)

# Тестирование нейронной сети

# Загрузить в список тестовый набор данных CSV-файла набора MNIST
test_data_file = open("MNIST/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# # Первый тест
# all_values_ts1 = test_data_list[0].split(',')
# # Вывести маркер
# print(all_values_ts1[0])
#
# # Вывод результатов нейронной сети
# print(network.query(numpy.asfarray(all_values_ts1[1:]) / 255.0 * 0.99) + 0.01)
#
# # Вывод изображения
# image_array = numpy.asfarray(all_values_ts1[1:]).reshape((28, 28))
# matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation="None")
# matplotlib.pyplot.show()

# Журнал оценок работы сети
scorecard = []

# Перебрать все записи в тестовом наборе данных
for record in test_data_list:
    all_values_ts = record.split(',')
    # Правильный ответ - первое значение
    correct_label = int(all_values_ts[0])
    # print(correct_label, "истинный маркер")
    # Масштабировать и сместить входные значения
    inputs = numpy.asfarray(all_values_ts[1:]) / 255.0 * 0.99 + 0.01
    # Опрос сети
    outputs = network.query(inputs)
    # Индекс наибольшего значения является маркерным значением
    label = numpy.argmax(outputs)
    # print(label, "ответ сети")
    # Добавление результата в журнал
    if label == correct_label:
        # В случае правильного ответа сети присоединить к списку значение 1
        scorecard.append(1)
    else:
        # В случае неправильного - 0
        scorecard.append(0)

# Вывод записей журнала
# print(scorecard)

# Расчет показателя эффективности в виде доли правильных ответов
scorecard_array = numpy.asarray(scorecard)
print("Эффективность равна", scorecard_array.sum() / scorecard_array.size)
