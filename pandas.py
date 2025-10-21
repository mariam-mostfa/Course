import pandas as pd

students_data = [
    {"Name": "Alice", "Age": 20, "Grade": "A", "Marks": 85},
    {"Name": "Bob", "Age": 22, "Grade": "B", "Marks": 78},
    {"Name": "Charlie", "Age": 19, "Grade": "A", "Marks": 92},
    {"Name": "David", "Age": 21, "Grade": "C", "Marks": 65},
    {"Name": "Eva", "Age": 20, "Grade": "B", "Marks": 74},
]
df = pd.DataFrame(students_data)
print(df)
print(df.head(3))
print(df[["Name", "Marks"]])
print(df.loc[df["Grade"] == "A"])
