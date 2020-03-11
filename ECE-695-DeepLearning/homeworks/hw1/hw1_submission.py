## ECE 695: HW1
## author: Rahul Deshmukh
## email: deshmuk5@purdue.edu

import random, string, numpy as np
random.seed(0)

#--------------------------- Class Definitions -----------------------------------#
class People:
    "base class"
    def __init__(self,first_names,middle_names,last_names,name_format='first_name_first'):
        self.first_names = first_names
        self.middle_names = middle_names
        self.last_names = last_names
        self.len = len(first_names)
        self.name_format = name_format
    
    def __iter__(self):
        self.count = 0 
        return self

    def __next__(self):
        if self.count >= self.len: raise StopIteration
        if self.name_format == 'first_name_first':
            print("%s %s %s"%(self.first_names[self.count],\
                               self.middle_names[self.count],\
                               self.last_names[self.count]),end=' ')
        elif  self.name_format == 'last_name_first':
            print("%s %s %s"%(self.last_names[self.count],\
                               self.first_names[self.count],\
                               self.middle_names[self.count]),end=' ')
        elif  self.name_format == 'last_name_with_comma_first':
            print("%s, %s %s"%(self.last_names[self.count],\
                               self.first_names[self.count],\
                               self.middle_names[self.count]),end=' ')
        else : print('unexpected name format!');raise StopIteration
        self.count +=1
    next = __next__ # for Py2 compatibility

    def __call__(self):
        sorted_last_names = sorted(self.last_names)
        for name in sorted_last_names: print(name)

class PeopleWithMoney(People):
    "subclass with money information"
    def __init__(self,first_names,middle_names,last_names,wealth,name_format='first_name_first'):
        super().__init__(first_names,middle_names,last_names,name_format)
        self.wealth = wealth

    def __iter__(self):
        People.__iter__(self)
        return self

    def __next__(self):
        People.__next__(self)
        print(self.wealth[self.count-1])
    next = __next__ # for Py2 compatibility

    def __call__(self):
        sorted_ids = np.argsort(self.wealth)
        for j in range(len(self.wealth)):
                i = sorted_ids[j]
                print("%s %s %s %d"%(self.first_names[i],\
                                     self.middle_names[i],\
                                     self.last_names[i],\
                                     self.wealth[i]))


#--------------------------- Main File -----------------------------------#

num_people = 10
name_size = 5

# Task 2
first_names= [ ''.join(random.choice(string.ascii_lowercase) for _ in range(name_size))  for i in range(num_people)]
middle_names= [ ''.join(random.choice(string.ascii_lowercase) for _ in range(name_size))  for i in range(num_people)]
last_names= [ ''.join(random.choice(string.ascii_lowercase) for _ in range(name_size))  for i in range(num_people)]

# Task 3        
people = People(first_names,middle_names,last_names)

# Task 4
for _ in people:print("");continue
print('')

# Task 5
name_format = {1:'first_name_first',2:'last_name_first',3:'last_name_with_comma_first'}
for i in range(1,len(name_format)):
        people.name_format=name_format[i+1]
        for _ in people:print("");continue
        print('')


# Task 6
people()
print('')

# Task 7
wealths = [random.randint(0,1000) for _ in range(num_people)]
people_with_wealth = PeopleWithMoney(first_names,middle_names,last_names,wealths)
for _ in people_with_wealth:continue
print('')

people_with_wealth() 
