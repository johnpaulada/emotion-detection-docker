def decide(value):
    if value is None:
            return Nothing()
    elif isinstance(value, list):
        return List(value)

    return Just(value)

# Maybe Monad

class Maybe(object):
    @staticmethod
    def of(value):
        if value is None:
            return Nothing()
        else:
            return Just(value)

class Nothing(object):
    def __init__(self):
        self.__value = None

    def map(self, f):
        return Nothing()
    
    def chain(self, f):
        return Nothing()
    
    def reduce(self, f, d=None):
        return Nothing()

    @property
    def value(self):
        return self.__value

class Just(object):
    def __init__(self, value):
        self.__value = value

    def map(self, f):
        return decide(f(self.__value))
    
    def chain(self, f):
        return decide(f(self.__value).value)

    @property
    def value(self):
        return self.__value

# List Monad

class List(object):
    def __init__(self, value):
        self.__value = value

    def map(self, f):
        return create_list_applyer(self, f, list_map)

    def chain(self, f):
        return create_list_applyer(self, f, list_chain)

    def concat(self, x):
        return List(self.__value + [x])

    def fold(self, f):
        return f(self.__value)

    def reduce(self, f, d=[]):
        reduced_value = d

        for value in self.__value:
            reduced_value = f(value, reduced_value)

        return decide(reduced_value)

    def filter(self, f):
        return List([v for v in self.__value if f(v)])

    @property
    def value(self):
        return self.__value

    @staticmethod
    def of(value):
        if value is None:
            return Nothing()
        else:
            return List(value)

def create_list_applyer(self, f, function_applyer):
    return List(function_applyer(f, self.value))

def create_list_function_applyer(evaluator):
    def list_applyer(f, l):
        return [evaluator(f, value, index) for index, value in enumerate(l)]

    return list_applyer

list_map = create_list_function_applyer(lambda f, v, i: f(v, i))
list_chain = create_list_function_applyer(lambda f, v, i: f(v, i).value)