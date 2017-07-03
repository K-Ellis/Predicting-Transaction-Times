d = {}
with open("../../../Data/prepare.txt", "r") as f:
#     content = f.readlines()
# content = [x.strip() for x in content]
# print(content)


# with open("file.txt") as f:
    for line in f:
        line = line.replace(":", "")
        (key, val) = line.split()
        d[key] = val
print(d)

