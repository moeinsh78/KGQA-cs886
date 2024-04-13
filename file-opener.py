from numpy import load


data = load('dataset/MetaQA/MetaQA-3/entity/kb_entity.npz')
lst = data.files
i = 0

for item in lst:
    if i > 0:
        break
    print(item)
    print(data[item])
    i += 1


print(len(lst[0]))
print(type(lst))
print(type(lst[0]))
