class Book:
    def __init__(self, title, author, isbn):
        self.title = title
        self.author = author
        self.__isbn = isbn
        self.available = True

    def display_info(self):
        print(self.title)
        print(self.author)
        print(self.__isbn)
        print(self.available)

    # ✅ دالة getter للحصول على الـ isbn
    def get_isbn(self):
        return self.__isbn

    # ✅ دالة setter لتغيير الـ isbn
    def set_isbn(self, new_isbn):
        self.__isbn = new_isbn


class Member:
    def __init__(self, name, membership_id):
        self.name = name
        self.__membership_id = membership_id
        self.borrow_books = []

    def borrow_book(self, book):
        if book.available:
            book.available = False
            self.borrow_books.append(book)
            print(self.name + " borrowed " + book.title)
        else:
            print(book.title + "not available")

    def return_book(self, book):
        if book in self.borrow_books:
            book.available = True
            self.borrow_books.remove(book)
            print(self.name + " return" + book.title)
        else:
            print(book.title + self.name)

    # ✅ دالة getter للحصول على رقم العضوية
    def get_membership_id(self):
        return self.__membership_id

    # ✅ دالة setter لتغيير رقم العضوية
    def set_membership_id(self, new_id):
        self.__membership_id = new_id


class StaffMember(Member):
    def __init__(self, name, membership_id, staff_id):
        super().__init__(name, membership_id)
        self.staff_id = staff_id

    def add_book(self, library, title, author, isbn):
        new_book = Book(title, author, isbn)
        library.books.append(new_book)
        print("Book added" + ",", title)


class Library:
    def __init__(self):
        self.books = []


# ✅ الكائنات والطباعة في نهاية الملف
book1 = Book("rr", "g", "12345")
member1 = Member("dv", "M001")

print(book1.get_isbn())
print(member1.get_membership_id())
