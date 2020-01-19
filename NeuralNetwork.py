import numpy
import scipy.special
import scipy.ndimage
import imageio
import matplotlib.pyplot
import os.path


# Работа с нейронной сетью на основе данных MNIST

# Тренировочный набор
# https://www.pjreddie.com/media/files/mnist_train.csv

# Тестовый набор
# https://www.pjreddie.com/media/files/mnist_test.csv

# 10 записей из тестового набора данных MNIST
# https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_test_10.csv

# 100 записей из тренировочного набора данных MNIST
# https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_train_100.csv

# Определение класса нейронной сети
class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learning_rate):
        # Количество узлов во входном, скрытом и выходном слоях
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Матрицы весовых коэффициентов связей wih(между входным и скрытым слоями)
        # и who(между скрытым и выходными слоями)
        # Создание матрицы весовых связей с помощью нормального распределения со стандартным отклонением
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # Использование сигмоиды в качестве функции активации
        self.activation_function = lambda x: scipy.special.expit(x)

        # Обратная функция активации
        self.inverse_activation_function = lambda x: scipy.special.logit(x)

        # Коэффициент обучения
        self.lr = learning_rate

    # Тренировка нейронной сети
    def train(self, inputs_list, targets_list):
        # Преобразовать список входных значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # Рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # Рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # Рассчитать ошибки выходного слоя
        # (ошибка = целевое значение - фактическое значение)
        output_errors = targets - final_outputs
        # Рассчитать ошибки скрытого слоя
        # (ошибки output_errors, распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах)
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Обновить весовые коэффициенты для связей между скрытым и выходным слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # Обновить веоовые коэффициенты для связей между входным и скрытым слоями
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

    def query(self, inputs_list):
        # Преобразовать список входных значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T

        # Рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # Рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    # Построение изображения по заданным итоговым данным
    def backquery(self, targets_list):
        # Транспонирование массива?????
        final_outputs = numpy.array(targets_list, ndmin=2).T

        # Вычисление сигнала, подаваемого на выходные узлы
        final_inputs = self.inverse_activation_function(final_outputs)

        # Вычисление сигнала, выходящего из скрытых узлов
        hidden_outputs = numpy.dot(self.who.T, final_inputs)

        # ????
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # Вычислить сигнал, подаваемый на скрытые узлы
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # Вычисление сигнала, выходящего из входных узлов
        inputs = numpy.dot(self.wih.T, hidden_inputs)

        # ????
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs

    def input_nodes(self):
        return self.inodes

    def hidden_nodes(self):
        return self.hnodes

    def output_nodes(self):
        return self.onodes


def image_test_network(image_file_name, network):
    label = int(image_file_name.split(".png")[0][-1])
    img_array = imageio.imread(image_file_name, as_gray=True)
    img_data = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01
    record = numpy.append(label, img_data)
    print(record[0])
    # Вывод результатов нейронной сети
    network_output = network.query(record[1:])
    print(network_output)
    print(numpy.argmax(network_output))

    # Вывод изображения
    image_array = numpy.asfarray(record[1:]).reshape((28, 28))
    matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation="None")
    matplotlib.pyplot.show()


def train_network(network):
    # Загрузить в список тестовый набор данных CSV-файла набора MNIST
    training_data_file = open("MNIST/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # Переменная epochs указывает, сколько раз тренировочный набор данных используется для тренировки сети
    epochs = 5
    for i in range(epochs):
        # Перебрать все записи в тренировочном наборе данных
        for record in training_data_list:
            # Получить список значений
            all_values_tr = record.split(',')
            # Масштабировать и сместить входные значения
            inputs = (numpy.asfarray(all_values_tr[1:]) / 255.0 * 0.99) + 0.01
            # Создать целевые выходные значения(все равны 0.01,
            # за исключением желаемого маркерного значения, равного 0.99)
            targets = numpy.zeros(network.output_nodes()) + 0.01
            targets[int(all_values_tr[0])] = 0.99
            network.train(inputs, targets)

            # Создать повернутые варианты цифр
            # Повернуть против часовой клетки
            inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), 10,
                                                                  cval=0.01, order=1, reshape=False)
            network.train(inputs_plusx_img.reshape(784), targets)

            # Повернуть по часовой стрелке
            inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), -10,
                                                                   cval=0.01, order=1, reshape=False)
            network.train(inputs_minusx_img.reshape(784), targets)


def test_network(network):
    # Тестирование нейронной сети

    # Загрузить в список тестовый набор данных CSV-файла набора MNIST
    test_data_file = open("MNIST/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

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

    # Расчет показателя эффективности в виде доли правильных ответов
    scorecard_array = numpy.asarray(scorecard)
    print("Эффективность равна", scorecard_array.sum() / scorecard_array.size)

    # Вывод записей журнала
    return scorecard


def one_test_network(network):
    # Первый тест
    test_data_file = open("MNIST/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    all_values_ts1 = test_data_list[0].split(',')
    # Вывести маркер
    print(all_values_ts1[0])

    # Вывод результатов нейронной сети
    print(network.query(numpy.asfarray(all_values_ts1[1:]) / 255.0 * 0.99) + 0.01)

    # Вывод изображения
    image_array = numpy.asfarray(all_values_ts1[1:]).reshape((28, 28))
    matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation="None")
    matplotlib.pyplot.show()


def inverse_network(network, label):
    # Создать выходной сигнал
    targets = numpy.zeros(network.output_nodes()) + 0.01
    targets[label] = 0.99
    print(targets)

    # Получить данные для изображения
    image_data = network.backquery(targets)

    # Вывод изображения
    matplotlib.pyplot.imshow(image_data.reshape(28, 28), cmap="Greys", interpolation="None")
    matplotlib.pyplot.show()


# Создать экземпляр нейронной сети
input_nodes = 784
output_nodes = 10
print("Введите количество скрытых узлов")
hidden_nodes = int(input())
print("Введите коэффициент обучения")
learning_rate = float(input())
network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# train_network(network)
# inverse_network(network, 3)
# image_test_network(os.path.join("My_Images", "5.png"))
# one_test_network(network)
# test_network(network)
