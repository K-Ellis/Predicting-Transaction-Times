# Read in parameters as a list if there are more than two items in a line. 
# Else, read the single value in as string as before.
def get_parameters(parameters):
    d = {}
    with open(parameters, "r") as f:
        for line in f:
            line = line.replace(":", "")
            line = line.replace(",", "")
            line = line.split()
            key = line.pop(0)
            if len(line) > 1:
                d[key] = line
            else:
                d[key] = line[0]
    return d

if __name__ == "__main__":
    parameters = "../../../Data/parameters.txt"	 # Parameters file
    d = get_parameters(parameters)
    print(type(d))
    print(d["list"][0])
    print(type(d["list"][0]))
    print(d["input_files"])
    print(d["input_file"])

