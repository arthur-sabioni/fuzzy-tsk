
import numpy as np
import decimal
import math


def function(x):
    return x*x

class FuzzyTsk:
    
    def generate_new_points(self):
        self.points = np.array([(x, self.f(x)) for x in np.random.uniform(-2, 2, self.n_points)])
    
    def __init__(self, function, n_points=1000, epocas=1000, min_x=-2, max_x=2, alpha=0.01):
        
        self.f = function
        self.n_points = n_points
        self.epocas = epocas
        self.min_x = min_x
        self.max_x = max_x
        self.alpha = alpha
        
        # gerar pontos
        self.points = []
        self.generate_new_points()
        
    def execute(self):
        
        # iniciar p1, p2 e q
        x1m = -2
        x1s = 1
        p1 = -2
        q1 = 0
        
        x2m = 2
        x2s = 1
        p2 = 2
        q2 =0
        
        somaErros = 0
        for epoca in range(0, self.epocas):
            # fazer inferencia de todos os pontos gerados
            for i,x in enumerate(self.points.T[0]):
                #Calcular w1 e w2 para calcular y de cada um
                w1 = math.exp(-1/2*math.pow((x-x1m)/x1s, 2))
                w2 = math.exp(-1/2*math.pow((x-x2m)/x2s, 2))
                
                w1n = w1/(w1 + w2)
                w2n = w2/(w1 + w2)
                
                y1 = p1*x + q1
                y2 = p2*x + q2
                
                y = w1n*y1 + w2n*y2
                yd = self.points[i][1]
                e = y - yd
                
                # calcular todos os parametros
                p1 = p1 - self.alpha*e*w1n*x
                p2 = p2 - self.alpha*e*w2n*x
                
                q1 = q1 - self.alpha*e*w1n
                q2 = q2 - self.alpha*e*w2n
                
                x1m = x1m - self.alpha*e*w2*(y1-y2)/(pow(w1+w2, 2))*x-x1m/pow(x1s,2)*math.exp(math.pow(-1/2*((x))/x1s,2))
                x2m = x2m - self.alpha*e*w1*(y1-y2)/(pow(w1+w2, 2))*x-x2m/pow(x1s,2)*math.exp(math.pow(-1/2*((x))/x2s,2))
            
                somaErros = somaErros + e
            # gerar novos pontos para a próxima iteração
            self.generate_new_points()
            
            # plotar a curva de tempos em tempos
            print(str(epoca) + ": " + str(somaErros))
        
if __name__ == "__main__":
    
    ft = FuzzyTsk(function, alpha=0.001)
    ft.execute()