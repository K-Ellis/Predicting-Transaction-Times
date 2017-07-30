# number_test_1 = [7,11,10]
# len(y_test_pred) = 100
# len(df) = 1000
# number_test_1/len(y_test_pred) = [7/len(y_test_pred),11/len(y_test_pred),10/len(y_test_pred)]
# number_test_1/len(y_test_pred) = [7/100,11/100,10/100]
# number_test_1/len(y_test_pred)*100 = [7%,11%,10%]
# np.mean(# number_test_1/len(y_test_pred)*100) = np.mean([7%,11%,10%])
# np.mean(# number_test_1/len(y_test_pred)*100) = np.mean([9.5%])

import numpy as np
len_y_test_pred = 100
number_test_96 = [99, 100, 100, 100, 100, 99, 95, 98, 100, 98]
np_number_test_96 = np.array(number_test_96)
pct = np.mean(np_number_test_96/len_y_test_pred*100)
std = np.std(np_number_test_96/len_y_test_pred*100)
print(pct)
print(std)
