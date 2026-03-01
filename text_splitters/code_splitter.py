from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """
class Car:
    def __init__(self, make, model, year):
      
        #Initialize the Car with specific attributes.
        self.make = make
        self.model = model
        self.year = year

# Creating an instance using the parameterized constructor
car = Car("Honda", "Civic", 2022)
print(car.make)
print(car.model)
print(car.year)
"""

code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 100,
    chunk_overlap =0
)

result = code_splitter.split_text(text)

print(result)