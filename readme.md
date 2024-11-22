# Bird Classifier


## Описание
Приложение, распознающее изображения домашних птиц (курица, петух, страус, утка, цыпленок, гусь, индюк). Обученные модели достигают точности более 85% на тестовом наборе.

## Файлы проекта
- `squeezenet_distill.pth`: веса обученной модели.
- `training.ipynb`: Jupyter-ноутбук с обучением модели.
- `bot.py`: код Telegram-бота для распознавания.
- `requirements.txt`: зависимости проекта.

## Запуск Telegram-бота
1. Установите зависимости:
    ```bash
    pip install -r requirements.txt
    ```
2. Установите токен Telegram API как переменную окружения:
    ```bash
    export TELEGRAM_BOT_TOKEN='<ваш_токен>'
    ```
3. Запустите бот:
    ```bash
    python bot/bot.py
    ```


## Воспроизведение обучения