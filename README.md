# Задание вступительного этапа Tinkoff Generation ML
## 'Антиплагиат'
Предлагалось написать утилиту для проверки двух Python-кодов на антиплагиат.
В самом простейшем случае можно было использовать расстояние [Левенштейна](https://en.wikipedia.org/wiki/Levenshtein_distance) для пред обработанных с помощью модуля `ast` кодов. 

Для каждого файла преобразование кода заключалось в:
- получении для данного кода ast ([Abstract Syntax Tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree))
- удалении комментариев (docstring для функций и классов), пустых функций
- замене исходных названий переменных, функций, классов и их аттрибутов на стандартизированные
- замене простого арифметического выражения на его результат
- получении python-кода для уже пред обработанного дерева

Дальше вычислялось расстояние Левинштейна для этих новых кодов. Полученное число нормализовалось на среднюю длину этих двух файлов.

## Usage

```
usage: compare.py [-h] f_in f_out

positional arguments:
  f_in        Input path for file
  f_out       Output path for the score file

options:
  -h, --help  show this help message and exit
```
