# AccountHack
Для использования приложения необходимо обучить NER BERT модель (например на основе приложенного NerTrain.txt файла), скрипт для обучения также приложен.

Обученную модель вместе со словарем для предобученного BERT на русском языке (можно загрузить с deeppavlov.ai) положить в папку "NER_mod_1" оставив в окружении все папки/файлы данного репозитория.

Можно использовать NER сеть через POST запросы к API (ajax: ['data'] = TEXT_TO_PREDICT)

В директории представлены также скрипты для построения линейных регрессий (от одного зависимого параметра) методом МНК
