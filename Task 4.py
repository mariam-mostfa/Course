# Task 1
def is_prime(n):
    if n < 2:
        return False

    for i in range(2, n):
        if n % i == 0:
            return False

    return True


print(is_prime(2))
print(is_prime(8))


# Task 2
def calculator(a, b, operation):
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            return "Sorry"
        return a / b
    else:
        return "This operation was not recognized."


print(calculator(6, 12, "add"))
print(calculator(25, 5, "divide"))
print(calculator(4, 0, "divide"))
print(calculator(3, 4, "sub"))
