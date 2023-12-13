class Language:
    def __init__(self,name):
        self.name= name
    def get_name(self):
        return self.name
    def message(self):
        print("Je suis " + self.name)

languages = [Language("Lena"), Language("Paul")]

for language in languages:
    language.message()

import numpy as np
import datetime 
test = np.random.randint(1,9)
print(test)
msg = "toi aussi ca va ? "
now = datetime.datetime.now()
print(msg)