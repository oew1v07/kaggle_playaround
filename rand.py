import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("out_scores_n_est_user_hot.csv", index_col="n_estimators")
df2 = pd.read_csv("out_scores_n_est_user_hot_sess.csv", index_col="n_estimators")
df3 = pd.read_csv("out_scores_n_est_user_nodates.csv", index_col="n_estimators")

mean1 = np.array(df.mean(axis=1))
mean2 = np.array(df2.mean(axis=1))
mean3 = np.array(df3.mean(axis=1))

n_est = np.array([  1,   8,  16,  24,  32,  40,  48,  55,  63,  71,  79,  87,  95,
       				102, 110, 118, 126, 134, 142, 150])

plt.style.use('ggplot')

fig = plt.figure(1)

l1 = plt.plot(n_est, mean1)
l2 = plt.plot(n_est, mean2)
l3 = plt.plot(n_est, mean3)

plt.legend(['Hot encoded user data', 'Hot encoded user + session data', 'User data without dates'], loc="best")
plt.xlabel("Number of decision tree estimators")
plt.ylabel("Normalised discounted cumulative gain")
plt.title("Effect of number of estimators on accuracy")
plt.savefig("n_estimators.svg", format="svg")
plt.savefig("n_estimators.pdf", format="pdf")

