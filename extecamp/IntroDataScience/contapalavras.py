
from collections import defaultdict



def map_function(text):
    for word in text.split():
        yield(word.lower(),1)


def reduce_function(key, values):
    return (key, sum(values))

text='Hello World MapReduce Hello'


group = defaultdict(list)

for key, value in map_function(text):
    group[key].append(value)

result = [reduce_function(key, values) for key, values in group.items()]
print(dict(result))

