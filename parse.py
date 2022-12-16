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
        count = {}
        for k in range(len(model._weights) + 1):
            count[k] = 0
        result = []
        
        if self.type == 'and':
            for child in self.childrens:
                temp = child.eval(model)
                k = 0
                for value in temp:
                    count[k] = int(count[k]) + (1 - value)**2
                    k += 1
        
            for k in count:
                result.append(1 - (count[k]/len(self.childrens))**0.5)
            
            return result
            
        for child in self.childrens:
            temp = child.eval(model)
            k = 0
            for value in temp:
                count[k] = int(count[k]) + value**2
                k += 1
        
        for k in count:
            result.append((count[k]/len(self.childrens))**0.5)
        return result

class Token(object):
    def __init__(self, value):
        self.value = value
    
    def eval(self, model): 
        result = []
        
        for k in model._weights:
            try:
                result.append(k[self.value])
            except KeyError:
                result.append(0)
        
        return result
    
    def __str__(self):
        return '  token: ' + self.value
            