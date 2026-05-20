import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

# ---------------------- قراءة البيانات وتدريب النموذج ----------------------
df = pd.read_csv("weather_dataset.csv")

# ترميز المتغيرات النصية
le_city = LabelEncoder()
le_weather = LabelEncoder()

df["City_Encoded"] = le_city.fit_transform(df["City"])
df["Weather_Encoded"] = le_weather.fit_transform(df["Weather_Condition"])
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month

# اختيار الميزات
features = [
    "Humidity_%",
    "Wind_Speed_kmph",
    "Rainfall_mm",
    "City_Encoded",
    "Weather_Encoded",
    "Month",
]
X = df[features]
y = df["Temperature_C"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# تقييم النموذج
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) * 100

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}%")


# ---------------------- واجهة المستخدم الاحترافية (Dark Mode) ----------------------
class WeatherApp:

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Temperature Prediction")

        # زيادة الأبعاد قليلاً لتعطي مساحة تنفسية للعناصر (جعلناها 500x550)
        self.window.geometry("500x550")
        self.window.configure(bg="#121212")

        # إعداد ستايل مخصص للقوائم المنسدلة (Combobox) لتتماشى مع النمط الداكن
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(
            "TCombobox",
            fieldbackground="#2C2C2C",
            background="#1E1E1E",
            foreground="white",
            arrowcolor="#FFA500",
        )

        self.model = model
        self.le_city = le_city
        self.le_weather = le_weather
        self.feature_names = features

        self.create_widgets()

    def create_widgets(self):
        # 1. عنوان النظام العلوي
        title = tk.Label(
            self.window,
            text="🌡️ Temperature Prediction System",
            bg="#1E1E1E",
            fg="#FFA500",  # تم تغيير اللون للبرتقالي ليعطي طابعاً حيوياً
            font=("Cairo", 14, "bold"),
            pady=8,
        )
        title.pack(pady=15, fill="x")

        # 2. الإطار الحاضن للبيانات
        input_frame = tk.LabelFrame(
            self.window,
            text=" Enter Weather Data ",
            bg="#1E1E1E",
            fg="#E0E0E0",  # تغيير الخط هنا للأبيض المائل للرمادي لراحة العين
            font=("Cairo", 12, "bold"),
            bd=1,
            relief="solid",
        )
        input_frame.pack(pady=10, padx=20, fill="both", expand=True)

        # تحسين توزيع الأبعاد داخل شبكة الإطار
        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=1)

        # التنسيق الموحد للنصوص والحقول الداكنة
        label_options = {
            "bg": "#1E1E1E",
            "fg": "#B3B3B3",
            "font": ("Cairo", 11, "bold"),
        }
        entry_options = {
            "bg": "#2C2C2C",
            "fg": "white",
            "insertbackground": "white",
            "relief": "flat",
            "font": ("Arial", 11),
        }

        # --- الرطوبة ---
        tk.Label(input_frame, text="Humidity (%) :", **label_options).grid(
            row=0, column=0, padx=15, pady=10, sticky="w"
        )
        self.humidity_entry = tk.Entry(input_frame, **entry_options)
        self.humidity_entry.grid(row=0, column=1, padx=15, pady=10, sticky="ew")

        # --- سرعة الرياح ---
        tk.Label(input_frame, text="Wind Speed (km/h) :", **label_options).grid(
            row=1, column=0, padx=15, pady=10, sticky="w"
        )
        self.wind_entry = tk.Entry(input_frame, **entry_options)
        self.wind_entry.grid(row=1, column=1, padx=15, pady=10, sticky="ew")

        # --- كمية المطر ---
        tk.Label(input_frame, text="Rainfall (mm) :", **label_options).grid(
            row=2, column=0, padx=15, pady=10, sticky="w"
        )
        self.rain_entry = tk.Entry(input_frame, **entry_options)
        self.rain_entry.grid(row=2, column=1, padx=15, pady=10, sticky="ew")

        # --- المدينة ---
        tk.Label(input_frame, text="City :", **label_options).grid(
            row=3, column=0, padx=15, pady=10, sticky="w"
        )
        self.city_combobox = ttk.Combobox(
            input_frame,
            values=list(self.le_city.classes_),
            state="readonly",
            style="TCombobox",
        )
        self.city_combobox.grid(row=3, column=1, padx=15, pady=10, sticky="ew")

        # --- حالة الطقس ---
        tk.Label(input_frame, text="Weather Condition :", **label_options).grid(
            row=4, column=0, padx=15, pady=10, sticky="w"
        )
        self.weather_combobox = ttk.Combobox(
            input_frame,
            values=list(self.le_weather.classes_),
            state="readonly",
            style="TCombobox",
        )
        self.weather_combobox.grid(row=4, column=1, padx=15, pady=10, sticky="ew")

        # --- الشهر ---
        tk.Label(input_frame, text="Month (1-12) :", **label_options).grid(
            row=5, column=0, padx=15, pady=10, sticky="w"
        )
        self.month_entry = tk.Entry(input_frame, **entry_options)
        self.month_entry.grid(row=5, column=1, padx=15, pady=10, sticky="ew")

        # 3. زر التنبؤ المتوهج باللون البرتقالي
        self.predict_btn = tk.Button(
            self.window,
            text="Predict Temperature",
            command=self.predict,
            padx=25,
            pady=8,
            bg="#FFA500",
            fg="black",
            activebackground="#FF8C00",
            activeforeground="black",
            font=("Cairo", 12, "bold"),
            relief="flat",
            cursor="hand2",
        )
        self.predict_btn.pack(pady=15)

        # 4. عرض النتيجة متناسبة مع النمط المظلم
        self.prediction_label = tk.Label(
            self.window,
            text="",
            font=("Cairo", 13, "bold"),
            fg="#FFD700",  # لون ذهبي ناصع للنتيجة ليجذب الانتباه
            bg="#121212",  # متطابق تماماً مع لون خلفية النافذة الأساسية
        )
        self.prediction_label.pack(pady=10)

    def predict(self):
        try:
            humidity = float(self.humidity_entry.get())
            wind_speed = float(self.wind_entry.get())
            rainfall = float(self.rain_entry.get())
            city = self.city_combobox.get()
            weather_cond = self.weather_combobox.get()
            month = int(self.month_entry.get())

            if month < 1 or month > 12:
                messagebox.showerror("Error", "Month must be between 1 and 12")
                return

            if not city:
                messagebox.showerror("Error", "Please select a city")
                return

            if not weather_cond:
                messagebox.showerror("Error", "Please select a weather condition")
                return

            city_encoded = self.le_city.transform([city])[0]
            weather_encoded = self.le_weather.transform([weather_cond])[0]

            input_data = np.array(
                [[humidity, wind_speed, rainfall, city_encoded, weather_encoded, month]]
            )

            predicted_temp = self.model.predict(input_data)[0]

            result_text = f"🔥 Predicted Temperature: {predicted_temp:.2f} °C"
            self.prediction_label.config(text=result_text)
            print(f"Prediction successful: {predicted_temp:.2f} °C")

        except ValueError:
            messagebox.showerror(
                "Error",
                "Please enter valid numbers in all fields (Humidity, Wind, Rainfall, Month)",
            )
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred:\n{str(e)}")

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = WeatherApp()
    app.run()
