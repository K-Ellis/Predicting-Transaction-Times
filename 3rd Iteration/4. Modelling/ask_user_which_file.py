def ask_user():
    answer = input('Press 1 to use the first COSMIC dataset. Press 2 (or any other key) to use the second COSMIC '
                   'dataset: ')
    if answer == "1":
        print("Using COSMIC 1:\n")
        return answer
    else:
        print("Using COSMIC 2:\n")
        return 2
