from math import sqrt
from data_collections import default_processor

class Expression(object):
    def __init__(self, type, lang='english'):
        self.childrens = []
        self.type = type
        self.lang = lang
        self._ident = ''
    
    def parse(self, query, index=0):
        
        while index < len(query):
            if query[index] == '(':
                expr = Expression('and', self.lang)
                self.childrens.append(expr)
                index = expr.parse(query, index + 1)
            elif query[index] == ')':
                return index + 1
            elif query[index] == '|' or query[index] == '&':
                index += 1
            else:
                token = Token(default_processor(query[index], self.lang)[0])
                self.childrens.append(token)
                index += 1
        
        return index
        
    def __str__(self):
        if self.type == 'and':
           self._ident += ' '
        string = self._ident + 'expression:\n' + self._ident + 'type: ' + self.type
        for child in self.childrens:
            string += '\n' + child.__str__()
        return string
    
    def eval(self, model):
        count = 0    
        result = []
        
        if self.type == 'and':
            for child in self.childrens:
                for doc in model.weights:
                    count += (1 - child.eval(model))**2
                result.append(1 - count**0.5)
                count = 0
            return result
            
        for child in self.childrens:
            for doc in model.weights:
                count += child.eval(model)**2
            result.append(count**0.5)
            count = 0
        return result

class Token(object):
    def __init__(self, value):
        self.value = value
    
    def eval(self, model):
        try:    
            return [value[self.value] for value in model.weights]
        except KeyError:
            return 0
    
    def __str__(self):
        return '  token: ' + self.value
            
    