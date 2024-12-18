import re

# Учитавамо текст из датотеке
input_file = "./data/СрпскаИменаЖенска.txt"  # Замени са именом твоје датотеке
output_file = "./data/имена_српска_женска.txt"

# Отвори улазну датотеку
with open(input_file, "r", encoding="utf-8") as file:
    content = file.read()

# Очисти текст:
# - Уклања све бројеве
# - Уклања вишеструке размака и размака око имена
# - Подели по зарезима, новим редовима или тачкама
names = re.split(r'[,\n\.]+', content)  # Раздваја по зарезима, новим редовима и тачкама
names = [re.sub(r'\d+', '', name).strip() for name in names]  # Уклања бројеве и размаке
names = [name for name in names if name]  # Уклања празне елементе

# Сачувај форматирна имена у новој датотеци
with open(output_file, "w", encoding="utf-8") as file:
    file.write("\n".join(names))

print(f"Имена су успешно форматирна и сачувана у {output_file}")
