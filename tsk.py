
import numpy as np


def function(x):
    return x*x

class FuzzyTsk:
    
    def generate_new_points(self):
        self.points = [(x, self.f(x)) for x in np.random.uniform(-2, 2, self.n_points)]
    
    def __init__(self, function, n_points=1000, epocas=1000, min_x=-2, max_x=2, alpha=0.01):
        
        self.f = function
        self.n_points = n_points
        self.epocas = epocas
        self.min_x = min_x
        self.max_x = max_x
        self.alpha = alpha
        
        self.points = []
        self.generate_new_points()
        
        print('test')
        
    def execute():
        
        # gerar pontos
        # gerar gaussianas
        # iniciar p1, p2 e q
        # fazer inferencia de todos os pontos gerados
        #   Calcular w1 e w2 para calcular y de cada um
        #   Pega a média do erro
        # calcular todos os parametros
        # gerar novos pontos para a próxima iteração
        # plotar a curva de tempos em tempos
        
        # Pode ser feito do modo acima, ou atualizar os valores para cada ponto
        pass
        
if __name__ == "__main__":
    
    ft = FuzzyTsk(function)
    ft.execute()