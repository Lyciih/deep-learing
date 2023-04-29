class Dog:
	def __init__(self, breed, name, age):
		self.breed = breed
		self.name = name
		self.age = age

	def bark(self):
		print("woof!")

def add_to_class(Class):
	def wrapper(obj):
		setattr(Class, obj.__name__, obj)
	return wrapper





dog = Dog("Labrador", "Max", 5)
dog.bark()
aog = Dog("Lab", "x", 5)

@add_to_class(Dog)
def run(self):
	print('Dog %s (age=%d) is running!'%(self.name,self.age))
dog.run()
aog.run()
