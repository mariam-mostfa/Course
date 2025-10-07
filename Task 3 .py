# Task 1
contacts = {"mariam": "018379646", "mostafa": "01746826", "omar": "01565468"}
print(contacts.keys())
search = input("Enter name to search: ")
if search in contacts:
    print(search, ":", contacts[search])
else:
    print(search, "not found in contacts")

# Task 2
students = [
    {"name": "mariam", "grades": [95, 85, 93]},
    {"name": "mhmd", "grades": [86, 75, 90]},
    {"name": "mirna", "grades": [90, 77, 80]},
]
for student in students:
    name = student["name"]
    grades = student["grades"]
    average_grade = sum(grades) / len(grades)
    print(name + ": " + str(average_grade))
