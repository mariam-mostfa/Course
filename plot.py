import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("weather_dataset.csv")
print(df.head())
plt.figure(figsize=(8, 6))
plt.plot(df["Date"], df["Temperature_C"], marker="o", color="pink")
plt.title("temperature variation over a week.")
plt.xlabel("the days of the week.")
plt.ylabel("the temperature in degrees Celsius.")
plt.show()
