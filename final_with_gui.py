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

# قراءه الداتا
df = pd.read_csv("weather_dataset.csv")
# معالجة الداتا
df["City_Encoded"] = LabelEncoder().fit_transform(df["City"])
df["Weather_Encoded"] = LabelEncoder().fit_transform(df["Weather_Condition"])
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month

# تقسيم الداتا
X = df[
    [
        "Humidity_%",
        "Wind_Speed_kmph",
        "Rainfall_mm",
        "City_Encoded",
        "Weather_Encoded",
        "Month",
    ]
]
y = df["Temperature_C"]
# بنقسم الداتا لتدريب و اختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# انشاء و تدريب نموذج الانحدار الخطي
model = LinearRegression()
model.fit(X_train, y_train)
# التنبؤ والتقييم
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) * 100

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# برسم صفحه رسم بياني ب عرض 10 و ارتفاع 4
plt.figure(figsize=(10, 4))
# صف واحد .. عمودين .. هبدا ب الرسمه الاولى على الشمال
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
#  كتبتهم مرتين علشان دي نقطه بتكون (100,100) مثلا
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--", color="red")
plt.xlabel("Actual values ")
plt.ylabel("Predictions")
plt.title(" Actual values vs Predictions")

# صف واحد .. عمودين .. هبدا ب الرسمه التانيه على اليمين
plt.subplot(1, 2, 2)
# حساب الفرق بين القيم الحقيقيه و القيم المتوقعه
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.7, color="green")

# axhline = Axis Horizontal Line
# y=0 zero means there are absolutely no errors.

plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predictions")
plt.ylabel("Residuals")
plt.title("The difference between actual and predicted")

# بيظبط المسافات بين الرسوم البيانيه
plt.tight_layout()
plt.show()


# GUI ***************************************
class WeatherApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Temperature prediction")
        self.window.geometry("400x400")  # زودت الارتفاع عشان تتسع لحقل الشهر
        self.window.configure(bg="firebrick")  # 🎨 لون خلفية النافذة
        # متغيرات لتخزين البيانات
        self.df = df
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        # حفظ قيمة r2 كمتغير في الكلاس
        # مقياس دقه النموذج
        self.r2 = r2

        # إنشاء الـ LabelEncoders للاستخدام في الواجهة
        # بستخدم الداله دي عشان احول اسماء المدن و حاله الطقس الى ارقام
        self.label_encoder_city = LabelEncoder().fit(df["City"])
        self.label_encoder_weather = LabelEncoder().fit(df["Weather_Condition"])
        # داله جواها كل الحقول و العناوين و الازرار
        self.create_widgets()

    ##################################################################
    def create_widgets(self):
        # عنوان الواجهه
        title = tk.Label(
            self.window,
            text="Temperature prediction",
            font=("Arial", 14),
            fg="red",
        )
        title.pack(pady=10)

        # إطار لإدخال البيانات
        input_frame = tk.LabelFrame(
            self.window, text="Enter the data", font=("Arial", 11), fg="red"
        )
        # padx مسافه يمين و شمال
        # pady مسافه فوق و تحت
        input_frame.pack(pady=10, padx=10)

        # إنشاء حقول الإدخال
        self.create_input_fields(input_frame)

        # زر التنبؤ
        self.predict_btn = tk.Button(
            self.window,
            text="Temperature prediction",
            # الحدث اللي هيتم تنفيذه
            command=self.predict,
            font=("Arial", 13),
            # حاله الزرار انه نشط و مستعد للاستخدام
            state="normal",
            fg="red",
        )
        self.predict_btn.pack(pady=10)

        # لعرض نتيجة التنبؤ
        self.prediction_label = tk.Label(
            self.window, text="", font=("Arial", 14, "bold"), fg="darkblue"
        )
        self.prediction_label.pack(pady=10)

    #################################################################################

    # """إنشاء حقول إدخال البيانات"""
    def create_input_fields(self, frame):
        # Humidity
        tk.Label(frame, text="Humidity % :", fg="darkblue").grid(
            row=0, column=0, padx=5, pady=5, sticky="e"
        )
        self.humidity_entry = tk.Entry(frame, fg="darkblue")
        self.humidity_entry.grid(row=0, column=1, padx=5, pady=5)

        # Wind speed
        tk.Label(frame, text="Wind speed (km/h) :", fg="darkblue").grid(
            row=1, column=0, padx=5, pady=5, sticky="e"
        )
        self.wind_entry = tk.Entry(frame, fg="darkblue")
        self.wind_entry.grid(row=1, column=1, padx=5, pady=5)

        # Rainfall
        tk.Label(frame, text="Rainfall (mm) :", fg="darkblue").grid(
            row=2, column=0, padx=5, pady=5, sticky="e"
        )
        self.rain_entry = tk.Entry(frame, fg="darkblue")
        self.rain_entry.grid(row=2, column=1, padx=5, pady=5)

        # City
        tk.Label(frame, text="City :", fg="darkblue").grid(
            row=3, column=0, padx=5, pady=5, sticky="e"
        )
        self.city_combobox = ttk.Combobox(
            frame, values=list(self.label_encoder_city.classes_), state="normal"
        )
        self.city_combobox.grid(row=3, column=1, padx=5, pady=5)
        self.city_combobox.set("")

        # Weather
        tk.Label(frame, text=" Weather :", fg="darkblue").grid(
            row=4, column=0, padx=5, pady=5, sticky="e"
        )
        self.weather_combobox = ttk.Combobox(
            frame, values=list(self.label_encoder_weather.classes_), state="normal"
        )
        self.weather_combobox.grid(row=4, column=1, padx=5, pady=5)
        self.weather_combobox.set("")

        # الشهر
        tk.Label(frame, text="Month (1-12) :", fg="darkblue").grid(
            row=5, column=0, padx=5, pady=5, sticky="e"
        )
        self.month_entry = tk.Entry(frame, fg="darkblue")
        self.month_entry.grid(row=5, column=1, padx=5, pady=5)

    ##############################################################################
    def predict(self):
        # """وظيفة التنبؤ بدرجة الحرارة"""
        try:
            # جلب البيانات من الحقول
            humidity = float(self.humidity_entry.get())
            wind_speed = float(self.wind_entry.get())
            rainfall = float(self.rain_entry.get())
            city = self.city_combobox.get()
            weather_condition = self.weather_combobox.get()
            month = int(self.month_entry.get())

            # التحقق من صحة المدخلات
            if month < 1 or month > 12:
                messagebox.showerror("خطأ", "يرجى إدخال شهر بين 1 و 12")
                return

            # تحويل المدينة وحالة الطقس إلى أرقام
            if city not in self.label_encoder_city.classes_:
                available_cities = ", ".join(self.label_encoder_city.classes_)
                messagebox.showerror(
                    "خطأ", f"المدينة غير موجودة. المدن المتاحة: {available_cities}"
                )
                return

            if weather_condition not in self.label_encoder_weather.classes_:
                available_weather = ", ".join(self.label_encoder_weather.classes_)
                messagebox.showerror(
                    "error",
                    f" No weather information available. Available statuses :{available_weather}",
                )
                return

            city_encoded = self.label_encoder_city.transform([city])[0]
            weather_encoded = self.label_encoder_weather.transform([weather_condition])[
                0
            ]

            ###############################################################################
            #  إنشاء مصفوفة المدخلات مكونه من صف واحد
            input_data = np.array(
                [[humidity, wind_speed, rainfall, city_encoded, weather_encoded, month]]
            )
            # التنبؤ بدرجة الحرارة
            predicted_temperature = self.model.predict(input_data)[0]
            # عرض النتيجة .. درجه الحراره بتبقى رقمين بعد العلامه و نعرض النتيجه في الواجهه
            result_text = f"Temperature prediction : {predicted_temperature: .2f} °C"
            self.prediction_label.config(text=result_text)

            print(f" The temperature was predicted : {predicted_temperature:.2f} °C")
            #  خطا من المستخدم انه يدخل حروف مكان الارقام
            # as e متغير بنخرن فيه الاخطاء error object
        except ValueError as e:
            messagebox.showerror(
                "error", "Please enter correct numerical values ​​in all fields."
            )
            #  خطا في البرنامج مش متوقع ومش عارفين نوعه زي النموذج مش متدرب او الملف مش موجود
            #  {str(e)} بنحول كائن الخطا لنص ونعرصه .. ده مش نص عادي
        except Exception as e:
            messagebox.showerror(
                "error", f"An error occurred in the prediction:{str(e)}"
            )

    ################################################################################
    # داله بتبدا تشغل التطبيق
    def run(self):
        """تشغيل الواجهة"""
        # الداله اللي بتخلي النافذه تفتح وتستني الاكشنز تحصل فيها ومهمه جداااا .. لو شلتها النافذه هتفتح وتقفل في نفس اللحظه
        self.window.mainloop()


# تشغيل التطبيق
if __name__ == "__main__":
    app = WeatherApp()
    app.run()
