# Установка одной Python-библиотеки

Endpoint для n8n tool:

```http
POST http://fastapi:8000/packages/install
```

## Body JSON

```json
{
  "package": "humanize"
}
```

## Что указать в n8n HTTP Request Tool

- Method: `POST`
- URL: `http://fastapi:8000/packages/install`
- Send Body: `true`
- Body Content Type: `JSON`
- Specify Body: `Using JSON`

JSON body для tool агента:

```json
{
  "package": "{{ $fromAI('package', 'pip package name to install, for example humanize or lightgbm', 'string') }}"
}
```

## Промпт для агента

```text
У тебя есть tool LIB_INSTALL для установки одной Python-библиотеки.

Используй его, если в Python-коде не хватает библиотеки.

Передавай только один параметр:
{
  "package": "pip_package_name"
}

Примеры:
{
  "package": "humanize"
}

{
  "package": "xgboost==2.0.3"
}

Не передавай массив библиотек. Если нужно несколько библиотек, вызывай tool несколько раз.
Не передавай строку "pip install ...".
Не передавай shell-команды, URL, пути к файлам или git-репозитории.

После success=true библиотеку можно сразу импортировать и использовать.
Если success=false, изучи install.stderr, определи причину ошибки и попробуй исправить имя пакета или версию. Не повторяй один и тот же ошибочный запрос бесконечно.
```

## Как работает

1. Сервер принимает одну строку `package`.
2. Проверяет, что строка похожа на обычный pip-пакет.
3. Запускает `python -m pip install --no-cache-dir <package>`.
4. Если установка успешна, добавляет пакет в `fastapi/requirements.txt`.
5. Если установка неуспешна, `requirements.txt` не меняется.
