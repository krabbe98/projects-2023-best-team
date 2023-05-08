from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import ipywidgets as widgets

class StackelbergSolver:
    def __init__(self):
        """ setup model """

        par = self.par = SimpleNamespace()

        par.a = 1 # demand for good 1
        par.b = 1 # demand for good 2
        par.X = 20 # demand for p=0
        par.c  = [0,0] # marginal cost
        
    def demand_function(self, q1, q2):
        par = self.par
        demand = par.X-par.a*q1-par.b*q2 # inverted demand function
        return demand

    def cost_function(self, q, c):
        return c*q # marginal cost times quantity

    def profits(self, c, q1, q2): 
        # income - expenditures
        return self.demand_function(q1,q2)*q1-self.cost_function(q1,c)
    
    def reaction(self, q2,c1):
        # Maaaaaaax profit
        
        responce =  optimize.minimize(lambda x: - self.profits(c1,x,q2), x0=0, method = 'SLSQP')
        return responce.x # best responce

    def fixed_point(self, q):
        par = self.par
        # the fixed point is the q that equals the reaction.
        return q-self.reaction(q,par.c[0])


    def solve_eq(self):
        par = self.par 
        initial_guess = np.array([0])
        # solve system of equations.
        res = optimize.minimize(lambda q1: -self.profits( par.c[0], q1,self.reaction(q1 , par.c[0]) ), initial_guess)
        q1 =res.x
        q2 = self.reaction(q1,par.c[1])
        return q1, q2



def plot_stackelberg(a=1, b=1, X=0, c1=0, c2=0):
    model = StackelbergSolver()
    # Update model parameters
    model.par.a = a
    model.par.b = b
    model.par.X = X
    model.par.c = [c1, c2]
    
    # x axis grid
    range_q1 = np.arange(0,21,0.1)
    # y axis grid
    range_q2 = np.zeros((range_q1.size,))
    
    for it, q1 in enumerate(range_q1):
        # find equilibrium quantities
        range_q2[it] = model.reaction(q1, c1)
    
    fig, ax = plt.subplots()
    ax.plot(range_q1, range_q2, label='Company 2')
    ax.set_xlabel('$q_1$')
    ax.set_ylabel('$q_2$')
    ax.set_title('Stackelberg Equilibrium Quantities')
    ax.legend()
    plt.show()

# Set up interactive sliders
def plot_stackelberg_interact():
    widgets.interact(
        plot_stackelberg,
        a=widgets.FloatSlider(description="a", min=1, max=5, step=0.25, value=1),
        b=widgets.FloatSlider(description="b", min=1, max=5, step=0.25, value=1),
        X=widgets.FloatSlider(description="X", min=1, max=50, step=0.5, value=20),
        c1=widgets.FloatSlider(description="c1", min=0, max=5, step=0.1, value=0),
        c2=widgets.FloatSlider(description="c2", min=0, max=5, step=0.1, value=0)
);




from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import ipywidgets as widgets

class StackelbergSolver2:
    def __init__(self):
        """ setup model """

        par = self.par = SimpleNamespace()

        par.a = 1 # demand for good 1
        par.b = 1 # demand for good 2
        par.X = 20 # demand for p=0
        par.c  = [0,0] # marginal cost
        par.t = [0,0.1]
        
    def demand_function(self, q1, q2):
        par = self.par
        demand = par.X-par.a*q1-par.b*q2 # inverted demand function
        return demand

    def cost_function(self, q, c):
        return c*q # marginal cost times quantity

    def profits1(self, c, q1, q2): 
        # income - expenditures
        return (self.demand_function(q1,q2)*q1-self.cost_function(q1,c))*(1-self.par.t[0])

    def profits2(self, c, q1, q2): 
        # income - expenditures
        return (self.demand_function(q1,q2)*q1-self.cost_function(q1,c))*(1-self.par.t[1])

    def reaction(self, q2,c1):
        # Maaaaaaax profit
        
        responce =  optimize.minimize(lambda x: - self.profits2(c1,x,q2), x0=0, method = 'SLSQP')
        return responce.x # best responce

    def fixed_point(self, q):
        par = self.par
        # the fixed point is the q that equals the reaction.
        return q-self.reaction(q,par.c[0])


    def solve_eq(self):
        par = self.par 
        initial_guess = np.array([0])
        # solve system of equations.
        res = optimize.minimize(lambda q1: -self.profits1( par.c[0], q1,self.reaction(q1 , par.c[0]) ), initial_guess)
        q1 =res.x
        q2 = self.reaction(q1,par.c[1])
        return q1, q2