"""
Решите задачу Separation (hw_fmnist) репозитории https://github.com/girafe-ai/ml-course/tree/25s_ml_trainings_3 в папке homeworks/hw01_classification.
После решения задачи классификации сгенерируйте посылку и загрузите в данную задачу submission_dict_task_1.json. Для этого файл hw**_data_dict.json из папки с задачей должен находиться в той же директории, что и ноутбук. Скрипт для автоматической загрузки доступен в ноутбуке, но при возникновении ошибок попробуйте скачать файл вручную.
Сгенерированный объект отправьте для сдачи в контест.

cм. https://colab.research.google.com/github/girafe-ai/ml-course/blob/25f_ml_trainings_4/homeworks/hw01_classification_and_attention/02_hw_fmnist_classification.ipynb
"""

import json
import os
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

# Добавим проверку для Windows
if __name__ == '__main__':
    # Установим device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Функции для получения предсказаний и точности
    def get_predictions(model, eval_data, step=10):
        predicted_labels = []
        model.eval()
        with torch.no_grad():
            for idx in range(0, len(eval_data), step):
                y_predicted = model(eval_data[idx: idx + step].to(device))
                predicted_labels.append(y_predicted.argmax(dim=1).cpu())

        predicted_labels = torch.cat(predicted_labels)
        predicted_labels = ",".join([str(x.item()) for x in list(predicted_labels)])
        return predicted_labels


    def get_accuracy(model, data_loader):
        predicted_labels = []
        real_labels = []
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                y_predicted = model(batch[0].to(device))
                predicted_labels.append(y_predicted.argmax(dim=1).cpu())
                real_labels.append(batch[1])

        predicted_labels = torch.cat(predicted_labels)
        real_labels = torch.cat(real_labels)
        accuracy_score = (predicted_labels == real_labels).type(torch.FloatTensor).mean()
        return accuracy_score


    # Загрузка данных
    print("Загрузка данных...")
    train_fmnist_data = FashionMNIST(
        ".", train=True, transform=torchvision.transforms.ToTensor(), download=True
    )
    test_fmnist_data = FashionMNIST(
        ".", train=False, transform=torchvision.transforms.ToTensor(), download=True
    )

    print(f"Размер тренировочной выборки: {len(train_fmnist_data)}")
    print(f"Размер тестовой выборки: {len(test_fmnist_data)}")

    # Устанавливаем num_workers=0 для избежания проблем с многопроцессорностью в Windows
    train_data_loader = DataLoader(
        train_fmnist_data, batch_size=32, shuffle=True, num_workers=0
    )

    test_data_loader = DataLoader(
        test_fmnist_data, batch_size=32, shuffle=False, num_workers=0
    )

    # Проверка размера входных данных
    sample_data, sample_label = next(iter(train_data_loader))
    print(f"Размер входного тензора: {sample_data[0].shape}")
    assert sample_data[0].shape == torch.Size([1, 28, 28]), "Неверный размер входных данных"


    # Определение модели
    class FashionMNISTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.25)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(-1, 64 * 7 * 7)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x


    # Создание экземпляра модели
    model_task_1 = FashionMNISTModel()
    print(f"Модель создана: {model_task_1}")
    model_task_1.to(device)
    print("Модель перенесена на устройство")

    # Проверка соответствия требованиям к модели
    test_input = torch.randn(1, 1, 28, 28).to(device)
    test_output = model_task_1(test_input)
    print(f"Размер выходных данных модели: {test_output.shape}")
    assert test_output.shape[1] == 10, "Модель должна выдавать 10 выходных значений"

    # Определение оптимизатора и функции потерь
    optimizer = torch.optim.Adam(model_task_1.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Обучение модели
    num_epochs = 10
    print("Начало обучения...")
    for epoch in range(num_epochs):
        model_task_1.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model_task_1(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 500 == 499:  # Печатаем каждые 500 батчей
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_data_loader)}], Loss: {running_loss / 500:.4f}')
                running_loss = 0.0

        print(f'Epoch {epoch + 1} completed')

    # Оценка точности
    train_acc_task_1 = get_accuracy(model_task_1, train_data_loader)
    test_acc_task_1 = get_accuracy(model_task_1, test_data_loader)
    print(f'Train Accuracy: {train_acc_task_1:.4f}')
    print(f'Test Accuracy: {test_acc_task_1:.4f}')

    # Проверка требований
    assert test_acc_task_1 >= 0.885, f"Test accuracy {test_acc_task_1} is below 0.885 threshold"
    assert train_acc_task_1 >= 0.905, f"Train accuracy {train_acc_task_1} is below 0.905 threshold"

    print("All requirements met!")

    # Генерация файла посылки
    if os.path.exists("hw_fmnist_data_dict.npy"):
        print("Найден файл hw_fmnist_data_dict.npy, создаем файл посылки...")
        loaded_data_dict = np.load("hw_fmnist_data_dict.npy", allow_pickle=True)

        submission_dict = {
            "train_predictions_task_1": get_predictions(model_task_1,
                                                        torch.FloatTensor(loaded_data_dict.item()["train"])),
            "test_predictions_task_1": get_predictions(model_task_1, torch.FloatTensor(loaded_data_dict.item()["test"]))
        }

        with open("submission_dict_fmnist_task_1.json", "w") as f:
            json.dump(submission_dict, f)
        print("Submission file created: submission_dict_fmnist_task_1.json")
    else:
        print("hw_fmnist_data_dict.npy not found. Skipping submission file creation.")

    print("Программа завершена успешно!")
