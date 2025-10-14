from Task5 import Book
from Task5 import Member
from Task5 import Library, StaffMember

book1 = Book("habta", "mhmd", 28172)
book2 = Book("el7ob w elhaya", "mostafa", 9577)
book1.display_info()


member1 = Member("omar", "333")
member2 = Member("Ali", "234")
member1.borrow_book(book1)
member2.return_book(book2)


lib = Library()
omar = StaffMember("ahmed", "22", "4")
omar.add_book(lib, "roah", "mariam", "3749749")
