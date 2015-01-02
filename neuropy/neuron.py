import numpy as np
from scipy.integrate import ode

"""Environ holds all the 'global' variables such as C, gK, gCa, etc."""
class Environ(object):

    def __init__(self, gK, gCa, gL, gKS, EK, ECa, EL, V1, V2, V3, V4, kc, Vc, C, Iext, de, ep):
        self.gK = gK
        self.gCa = gCa
        self.gL = gL
        self.gKS = gKS
        self.EK = EK
        self.ECa = ECa
        self.EL = EL
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3
        self.V4 = V4
        self.kc = kc
        self.Vc = Vc
        self.C = C
        self.Iext = Iext
        self.ep = ep
        self.de = de

class Model(object):
    def __init__(self,env):
        self.env = env

    def nInf(self,V):
        return 0.5*(1+np.tanh((V-self.env.V1)/self.env.V2))

    def mInf(self,V):
        return 0.5*(1+np.tanh((V-self.env.V3)/self.env.V4))

    def tm(self,V):
        return np.sech((V-self.env.V3)/(2*self.env.V4))

    def cInf(self,V):
        return 0.5*(1+tanh(self.env.kc*(V-Vc)))

    def tc(self,V):
        return np.sech((V-self.env.Vc)/(2*self.env.V4))

    def dyn(self,t,x):
        [V, m, c] = x

        ICa = self.env.gCa*self.mInf(V)*(V-self.env.ECa)
        IK = self.env.gK*m*(V-self.env.EK)
        IKS = self.env.gKS*c*(V-self.env.EK)
        IL = self.env.gL*(V-self.env.EL) #EK in paper
        
        Vd = -1./self.env.C*(ICa + IK + IL + IKS) + self.env.Iext/self.env.C
        md = self.env.ep*(mInf(V)-m)/self.tm(V)
        cd = self.env.de*(cInf(V)-c)/self.tc(V)
        
        return [Vd, md, cd]


"""Neuron is an object that contains the model for V, w and c"""

class Neuron(object):
    
    def __init__(self,env,model):
        self.env = env
        self.model = model
    def dyn(self,t,x):
        xd = self.model.dyn(t,x)
        return xd
    
    def sym(self,t0,x0,tmax,dt):
        r = ode(self.dyn).set_integrator('dopri5')
        r.set_initial_value(x0,t0)
        while r.successful() and r.t < tmax:
            r.integrate(r.t+dt)
            print("%g %g" %(r.t, r.y))
