# # 1
temp = int(input("Enter temperatrue"))
if temp >= 30:
    print("it is a hot dey . stay hydrated !")
elif temp >= 20:
    print("it is a warm . enjoy the weather !")
elif temp >= 10:
    print("it is a cool day . wear a jacket !")
else:
    print("it is a cold day . stay worm ! ")

# 2
for x in range(1, 21):
    if x % 3 == 0:
        print(x)
