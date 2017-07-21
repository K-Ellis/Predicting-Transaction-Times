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
	print(d)
	print(d["list"][0])