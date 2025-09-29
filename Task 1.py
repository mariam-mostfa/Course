shirt = 500
dress = 1000
skirt = 700

my_money = 2200

total_cost = shirt + dress + skirt

if total_cost <= my_money:
    print("true that is enough")
else:
    print("sorry")

subtraction = my_money - total_cost
if subtraction > 0:
    print("very good")
elif subtraction < 0:
    print("very bad")
else:
    print("exactly")
