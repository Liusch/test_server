from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Загрузка модели и токенизатора
tokenizer = transformers.BertTokenizer(vocab_file='H://python/test_project/vocab.txt')
config = transformers.BertConfig.from_json_file('H://python/test_project/config.json')
model = transformers.BertModel.from_pretrained('H://python/test_project/rubert_model.bin', config=config)

# Загрузка датасета твитов
df_tweets = pd.read_csv('H://python/test_project/tweets.csv')
df_tweets = df_tweets.reset_index(drop=True)

# Преобразуем текст в номера токенов
tokenized = df_tweets['text'].apply(
    lambda x: tokenizer.encode(x, add_special_tokens=True))

# Сделаем равными длины исходных текстов в корпусе с помощью паддингов
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)
padded = np.array([i + [0]*(max_len - len(i)) for i in tokenized.values])
attention_mask = np.where(padded != 0, 1, 0)

# Разобъём на батчи и преобразуем в тензоры, на выходе получим эмбединги
batch_size = 100
embeddings = []
for i in range(padded.shape[0] // batch_size):
    batch = torch.LongTensor(padded[batch_size*i:batch_size*(i+1)]) 
    attention_mask_batch = torch.LongTensor(attention_mask[batch_size*i:batch_size*(i+1)])
    
    with torch.no_grad():
        batch_embeddings = model(batch, attention_mask=attention_mask_batch)
    
    embeddings.append(batch_embeddings[0][:,0,:].numpy())

# Сформируем фичи и таргеты
features = np.concatenate(embeddings)
target = df_tweets['positive']

# Загрузка и обучение модели логистической регрессии
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(features, target)


# Обработчик POST-запроса
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Получение данных из POST-запроса
    tweet = data['tweet']  # Получение фразы из данных

    # Проверка длины сообщения
    if len(tweet) > max_len:
        result = {'Ошибка': 'Превышено максимальное количество символов'}
        return jsonify(result)
    if not tweet:
        result = {'Ошибка': 'Отправлено пустое сообщение'}
        return jsonify(result)

    # Токенизация и подготовка данных
    tokenized_tweet = tokenizer.encode(tweet, add_special_tokens=True)
    padded_tweet = np.array([tokenized_tweet + [0] * (max_len - len(tokenized_tweet))])
    attention_mask_tweet = np.where(padded_tweet != 0, 1, 0)

    # Преобразование данных в тензоры
    tweet_tensor = torch.LongTensor(np.array(padded_tweet))
    attention_mask_tensor = torch.LongTensor(np.array(attention_mask_tweet))

    # Получение эмбеддингов
    with torch.no_grad():
        embeddings = model(tweet_tensor, attention_mask=attention_mask_tensor)[0][:, 0, :].numpy()

    # Классификация эмбеддингов с помощью модели логистической регрессии
    prediction = model_lr.predict(embeddings)

    # Возвращение результата в формате JSON
    if int(prediction[0]) == 0:
        result = {'Твит': tweet, 'Предсказание': 'Текст имеет отрицательный или нейтральный оттенок'}
    else:
        result = {'Твит': tweet, 'Предсказание': 'Текст имеет положительный оттенок'}
    return jsonify(result)

if __name__ == '__main__':
    # Запуск сервера Flask
    app.run(debug=False)