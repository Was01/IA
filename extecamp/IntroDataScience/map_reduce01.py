from itertools import chain
from collections import defaultdict
from pathlib import Path

# Estrutura do Exemplo
# 0. Definição das funções
# 1. Leitura do Arquivo CSV
# 2. Fase de Mapeamento (Map)
# 3. Fase de Shuffle and Sort
# 4. Fase de Redução (Reduce)
# 5. Apresentar resultados

# 0. Define funções utilizadas no programa
def read_data(file_path):
    """
    Lê dados de um arquivo CSV e retorna uma lista de linhas, 
    desconsiderando o cabeçalho
    """
    with open(file_path, mode='r') as file:
        lines = file.readlines()

    # Remove cabeçalho, se houver
    if lines[0].startswith('disciplina'):
        del lines[0]
    return lines

def map_function(lines):
    """
    Função de mapeamento que recebe uma lista de linhas brutas e retorna 
    um gerador de pares (categoria, valor).
    """
    for line in lines:
        if line.strip():  # Ignorar linhas vazias
            parts = line.strip().split(',')
            category = parts[0]
            value = int(parts[1])
            yield (category, value)
        
def reduce_function(key, values):
    """
    Função de redução que recebe uma chave (categoria) e uma lista de 
    valores, e retorna a média dos valores.
    """
    return (key, sum(values) / len(values))

# Caminho para o arquivo CSV
print(Path.cwd())

# Caminho para o arquivo de texto
file_path = 'notas.csv'

# 1. Ler os dados do arquivo
data = read_data(file_path)
print(f'Entrada: {data}')
# Saída esperada: ['math,90\n', 'math,80\n', 'english,85\n', 
#                  'math,70\n', 'english,75']

# 2. Aplica a função de mapeamento a cada registro, combinando os resultados
mapped = list(chain(*[map_function([record]) for record in data]))
# Explicação da linha acima:
# a. List Comprehension: [map_function([record]) for record in data]
#    - Cria lista chamando 'map_function' para cada 'record' em 'data'
#    - Cada chamada map_function([record]) retorna um gerador (iterador)
#      de pares chave-valor
# b. Desempacotamento: *[map_function([record]) for record in data]
#    - Operador * desempacota a lista de geradores, passando cada um 
#      como um argumento separado para 'itertools.chain'.
# c. itertools.chain: itertools.chain(*[map_function([record]) for record in data])
#    - Combina vários iteradores (geradores) em um único que produz os elementos em sequência
# d. Conversão para Lista: list(itertools.chain(*[map_function([record]) for record in data]))
#    - list() converte o iterador resultante de itertools.chain em uma lista

print(f'Mapeados: {mapped}')
# Saída esperada: [('math', 90), ('math', 80), ('english', 85), 
#                  ('math', 70), ('english', 75)]

# 3. Agrupar os pares chave-valor por chave (shuffle & sort)
grouped = defaultdict(list)
for key, value in mapped:
    grouped[key].append(value)

print(f'Agrupados: {dict(grouped)}')
# Saída esperada: {'math': [90, 80, 70], 'english': [85, 75]}

# 4. Aplicar a função de redução a cada grupo de pares chave-valor
reduced = \
    [reduce_function(key, values) for key, values in grouped.items()]

# 5. Mostrar resultado final
print(f'Resultados: {reduced}')
# Saída esperada: [('math', 80.0), ('english', 80.0)]
