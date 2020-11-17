import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')


plt.plot(df["reynolds"][:7], df["expected_friction"][:7], 'y')
plt.plot(df["reynolds"][11:], df["expected_friction"][11:], 'g')
plt.plot(df["reynolds"], df["actual_friction"], 'k')
plt.plot(df["reynolds"], df["actual_friction"] + df["uncertainty"], "r", alpha=0.4)
plt.plot(df["reynolds"], df["actual_friction"] - df["uncertainty"], "r", alpha=0.4, label='_nolegend_')

plt.legend(["Expected Laminar Friction Factor", "Expected Turbulent Friction Factor", "Calculated Friction Factor", "Calculated Friction Factor Uncertainty Bound"])
plt.title("Friction Factor vs Reynolds Number")
plt.xlabel("Reynolds Number")
plt.ylabel("Friction Factor")
plt.show()