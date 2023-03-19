
import numpy as np
import random
import math

import matplotlib.pyplot as plt

def function(x):
    return x*x


def plot_2d(results: np.array, expected = [], labels=[''], title='', block=False):

    plt.figure()

    plt.plot(results.T[0], results.T[1], '-b', label=labels[0])
    if len(expected):
        plt.plot(expected.T[0], expected.T[1], '-r', label=labels[1])

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show(block=block)


class FuzzyTsk:
    
    def shuffle_points(self):
        self.points_shuffled = np.random.permutation(self.points)
    
    def __init__(self, function, step=0.01, epocas=1000, min_x=-2, max_x=2, alpha=0.01):
        
        self.f = function
        self.step = step
        self.epocas = epocas
        self.min_x = min_x
        self.max_x = max_x
        self.alpha = alpha
        
        # gerar pontos
        self.points = np.array([(x, self.f(x)) for x in np.arange(-2,2,self.step)])
        self.shuffle_points()
        
    def extimate_y(self, x1m, x1s, p1, q1, x2m, x2s, p2, q2, x, return_all=True):
        #Calcular w1 e w2 para calcular y de cada um
        w1 = math.exp(-1/2*math.pow((x-x1m)/x1s, 2))
        w2 = math.exp(-1/2*math.pow((x-x2m)/x2s, 2))
        
        w1n = w1/(w1 + w2)
        w2n = w2/(w1 + w2)
        
        y1 = p1*x + q1
        y2 = p2*x + q2
        
        y = w1n*y1 + w2n*y2
        
        if return_all:
            return w1, w2, w1n, w2n, y1, y2, y
        return y
        
    def execute(self):
        
        x1m = -2
        x1s = 1
        p1 = -2
        q1 = 0
        
        x2m = 2
        x2s = 1
        p2 = 2
        q2 = 0
                
        somaErros = 0
        listErros = []
        for epoca in range(0, self.epocas):
                
            # plotar a curva de tempos em tempos
            if not epoca or epoca % (self.epocas//10) == 0 or epoca == self.epocas - 1:
                extimated_points = np.array([(x, self.extimate_y(x1m, x1s, p1, 
                                                                q1, x2m, x2s, 
                                                                p2, q2, x, return_all=False)) for x in np.arange(-2,2,self.step)])
                plot_2d(extimated_points, self.points, ['estimado', 'esperado'],
                        'Após {} épocas - Erro = {}'.format(epoca, round(somaErros,2)))
             
            somaErros = 0

            # fazer inferencia de todos os pontos gerados
            for i,x in enumerate(self.points.T[0]):
                
                w1, w2, w1n, w2n, y1, y2, y = self.extimate_y(x1m, x1s, p1, q1, x2m, x2s, p2, q2, x)
                yd = self.points[i][1]
                e = y - yd
                
                # calcular todos os parametros
                p1 = p1 - self.alpha*e*w1n*x
                p2 = p2 - self.alpha*e*w2n*x
                
                q1 = q1 - self.alpha*e*w1n
                q2 = q2 - self.alpha*e*w2n
                
                x1m = x1m - self.alpha*e*w2*(y1-y2)/(pow(w1+w2, 2))*(x-x1m)/pow(x1s,2)*math.exp(-1/2*math.pow((x-x1m)/x1s,2))
                x2m = x2m - self.alpha*e*w1*(y2-y1)/(pow(w1+w2, 2))*(x-x2m)/pow(x2s,2)*math.exp(-1/2*math.pow((x-x2m)/x2s,2))
                
                x1s = x1s - self.alpha*e*w2*(y1-y2)/(pow(w1+w2, 2))*pow(x-x1m,2)/pow(x1s,3)*math.exp(-1/2*math.pow((x-x1m)/x1s,2))
                x2s = x2s - self.alpha*e*w1*(y2-y1)/(pow(w1+w2, 2))*pow(x-x2m,2)/pow(x2s,3)*math.exp(-1/2*math.pow((x-x2m)/x2s,2))
            
                somaErros = somaErros + pow(e, 2)
            
            # gerar novos pontos para a próxima iteração
            self.shuffle_points()
            
            listErros.append((int(epoca), somaErros))
            print(str(epoca) + ": " + str(somaErros))
            
        plot_2d(np.array(listErros), labels=['erro'], title='Erro por época', block=True)
        
if __name__ == "__main__":
    
    ft = FuzzyTsk(function, alpha=0.01, epocas=1000)
    ft.execute()