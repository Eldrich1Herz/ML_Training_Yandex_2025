'''
A. Attention implementation
Полное решение
Ограничение времени	20 секунд
Ограничение памяти	256 Мб
Ввод	стандартный ввод или no input
Вывод	стандартный вывод или tests.log

Выполните задания по имплементации Multiplicative и Additive attention, указанные в ноутбуке. После решения задачи перенесите код соответствующих функций в файл template_p01.py (доступен в репозитории) и сдайте его в контест.
Обращаем ваше внимание: вердикт ОК означает лишь то, что код запустился. Корректность выполнения задания оценивается баллами.
Полученные баллы, как и результаты прохождения различных тестов, можно увидеть по ссылке на "отчет" о посылке. Если вы видите баллы за задачу в формате 0.999999999, это нормально, в таком случае применяется округление по математическим правилам (т.е. оценка 1.0)
Максимальный балл за задачу: 1.0

cм. https://colab.research.google.com/github/girafe-ai/ml-course/blob/25f_ml_trainings_4/homeworks/hw01_classification_and_attention/01_attention.ipynb#scrollTo=yoYa0SSBWLOa
'''

import numpy as np

def softmax(vector):
    '''
    vector: np.array of shape (n, m)
    
    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    
    Математическая формула softmax для вектора x:
    softmax(x_i) = exp(x_i) / Σ_j(exp(x_j))
    
    Для численной стабильности вычитаем максимальное значение:
    softmax(x_i) = exp(x_i - max(x)) / Σ_j(exp(x_j - max(x)))
    '''
    nice_vector = vector - vector.max()  # Стабилизация численных значений
    exp_vector = np.exp(nice_vector)     # Экспоненцирование каждого элемента
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]  # Сумма по строкам
    softmax_ = exp_vector / exp_denominator  # Нормализация
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    
    Математическая формула multiplicative attention:
    e_i = s^T * W_mult * h_i
    α_i = softmax(e_i)
    context = Σ_i(α_i * h_i)
    
    Где:
    - s: состояние декодера (decoder_hidden_state)
    - h_i: i-ое состояние энкодера
    - W_mult: матрица весов
    - e_i: attention score для i-ого состояния
    - α_i: вес внимания для i-ого состояния
    - context: итоговый вектор контекста
    '''
    # Вычисляем attention scores: s^T * W_mult * H
    # decoder_hidden_state.T: (1, n_features_dec)
    # W_mult: (n_features_dec, n_features_enc)
    # encoder_hidden_states: (n_features_enc, n_states)
    # Результат: (1, n_states)
    scores = np.dot(np.dot(decoder_hidden_state.T, W_mult), encoder_hidden_states)
    
    # Применяем softmax для получения весов внимания
    # scores: (1, n_states) → attention_weights: (1, n_states)
    attention_weights = softmax(scores)
    
    # Вычисляем взвешенную сумму состояний энкодера
    # encoder_hidden_states: (n_features_enc, n_states)
    # attention_weights.T: (n_states, 1)
    # Результат: (n_features_enc, 1)
    attention_vector = np.dot(encoder_hidden_states, attention_weights.T)
    
    return attention_vector

def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    
    Математическая формула additive attention (Bahdanau attention):
    e_i = v^T * tanh(W_enc * h_i + W_dec * s)
    α_i = softmax(e_i)
    context = Σ_i(α_i * h_i)
    
    Где:
    - s: состояние декодера (decoder_hidden_state)
    - h_i: i-ое состояние энкодера
    - W_enc, W_dec: матрицы весов для проекции
    - v: вектор весов для получения скалярного значения
    - e_i: attention score для i-ого состояния
    - α_i: вес внимания для i-ого состояния
    - context: итоговый вектор контекста
    '''
    # Проецируем состояния энкодера: W_add_enc * H
    # W_add_enc: (n_features_int, n_features_enc)
    # encoder_hidden_states: (n_features_enc, n_states)
    # Результат: (n_features_int, n_states)
    enc_proj = np.dot(W_add_enc, encoder_hidden_states)
    
    # Проецируем состояние декодера: W_add_dec * s
    # W_add_dec: (n_features_int, n_features_dec)
    # decoder_hidden_state: (n_features_dec, 1)
    # Результат: (n_features_int, 1)
    dec_proj = np.dot(W_add_dec, decoder_hidden_state)
    
    # Складываем проекции и применяем нелинейность tanh
    # enc_proj: (n_features_int, n_states)
    # dec_proj: (n_features_int, 1) → broadcasting до (n_features_int, n_states)
    # Результат: (n_features_int, n_states)
    combined = np.tanh(enc_proj + dec_proj)
    
    # Вычисляем attention scores: v^T * combined
    # v_add.T: (1, n_features_int)
    # combined: (n_features_int, n_states)
    # Результат: (1, n_states)
    scores = np.dot(v_add.T, combined)
    
    # Применяем softmax для получения весов внимания
    # scores: (1, n_states) → attention_weights: (1, n_states)
    attention_weights = softmax(scores)
    
    # Вычисляем взвешенную сумму состояний энкодера
    # encoder_hidden_states: (n_features_enc, n_states)
    # attention_weights.T: (n_states, 1)
    # Результат: (n_features_enc, 1)
    attention_vector = np.dot(encoder_hidden_states, attention_weights.T)
    
    return attention_vector
