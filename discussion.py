from utils import import_data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


X_train, y_train, X_test = import_data()

std = StandardScaler()
X_train = pd.DataFrame(std.fit_transform(X_train))

# split X dataset where y=0 from y=1
X_0 = X_train[y_train == 0]
X_1 = X_train[y_train == 1]

# mean
mean_0 = X_0.mean(axis=0)
mean_1 = X_1.mean(axis=0)

# variation
var_0 = X_0.var(axis=0)
var_1 = X_1.var(axis=0)

# median
med_0 = X_0.median(axis=0)
med_1 = X_1.median(axis=0)

# random sample, for comparison
sample = X_train.sample(20000, axis=0)
mean_sample = sample.mean()

# show mean distribution
plt.hist(mean_0)
plt.hist(mean_1, bins=50)
plt.show()

# delta mean
mean_delta = abs(abs(mean_1) - abs(mean_0))
mean_sample_delta = abs(abs(mean_sample) - abs(mean_0))

# shows that random sample delta mean <0.025, therefore keep columns with a mean spread larger than that
# are statistically significant
n, bins, patches = plt.hist(mean_delta, bins=25)
plt.hist(mean_sample_delta, bins=bins)
plt.show()

# shows that random sample delta variance <1.03, therefore keep columns with a variance spread larger than that
plt.hist(var_0)
plt.hist(var_1, bins=50)
plt.show()

n, bins, patches = plt.hist(var_1, bins=25)
plt.hist(sample.var(), bins=bins)
plt.show()

# median
plt.hist(med_0)
plt.hist(med_1, bins=50)
plt.show()

# delta mean
med_delta = abs(abs(med_1) - abs(med_0))
med_sample_delta = abs(abs(mean_sample) - abs(med_0))

# shows that random sample delta median <0.075, therefore keep columns with a mean spread larger than that
# are statistically significant
n, bins, patches = plt.hist(med_delta, bins=25)
plt.hist(med_sample_delta, bins=bins)
plt.show()




