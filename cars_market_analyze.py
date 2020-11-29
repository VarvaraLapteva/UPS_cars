#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def prepare_data(data_file_name, test_size):
    df = pd.read_csv(data_file_name, sep=',')
    # Перепишем датафрейм, чтобы столбец car был на первом месте. Особого смысла в этом нет,
    # но так привычней :)
    cols = list(df)
    cols.insert(0, cols.pop(cols.index('car')))
    df = df.loc[:, cols]
    # Данные распределения по целевому признаку: вывести таблицу и гистограмму
    print(df["car"].value_counts())
    df["car"].value_counts().plot(kind="bar")
    plt.show()

    # конвертируем приемлемость автомобиля в числовое значение в соответствии со
    # словарём car_acc_dict: чем больше машина подходит, тем больше числовое значение
    car_acc_dict = {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}
    for tmp_key in car_acc_dict:
        df['car'].replace(tmp_key, car_acc_dict[tmp_key], inplace=True)

    # Конвертируем категориальные признаки в цифру, используя one-hot кодирование
    cols_to_transform = cols.copy()
    cols_to_transform.remove("car")
    df = pd.get_dummies(df, columns=cols_to_transform)

    # Теперь надо разделить конвертированные данные
    # на два набора - один для тренировки (большой)
    # и один для тестирования нейросети (поменьше)
    cols_to_transform = list(df)
    cols_to_transform.remove("car")
    x = df[cols_to_transform]
    y = df["car"]
    return train_test_split(x, y, test_size=test_size)


def test_samples():
    """Функция возвращает данные для четырех видов приемлемости:
    совсем никакой, так себе, хороший и классный"""
    return np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                     [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                     [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                     [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]])


if __name__ == "__main__":
    CARS_DATA_FILE = "cars/cars_dataset.csv"
    test_size = 0.05  # размер тестовой базы по отношению к общей базе: чем меньше, тем выше
    # точность предсказания
    # Подготавливаем данные свойств автомобилей для sklearn
    x_train, x_test, y_train, y_test = prepare_data(CARS_DATA_FILE, test_size)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    # Тренируем нейросеть
    clf.fit(x_train, y_train)

    print("\nТочность предсказаний: {0:f}\n".format(clf.score(x_test, y_test)))

    # Пробуем запустить нейросеть на четырех пробных наборах
    predictions = clf.predict(test_samples())

    print("Предсказания приемлемости автомобилей: {}\n".format(predictions))
