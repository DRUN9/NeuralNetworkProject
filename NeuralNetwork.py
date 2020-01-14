import numpy
import scipy.special


# Определение класса нейронной сети
class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learning_rate):
        # Количество узлов во входном, скрытом и выходном слоях
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Матрицы весовых коэффициентов связей wih(между входным и скрытым слоями)
        # и who(между скрытым и выходными слоями)
        # self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        # self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5

        # Еще один вариант создания матриц весовых связей
        # (с помощью нормального распределения со стандартным отклонением)
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # Использование сигмоиды в качестве функции активации
        self.activation_function = lambda x: scipy.special.expit(x)

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
