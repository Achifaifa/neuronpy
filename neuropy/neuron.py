import numpy as np
from math import tanh,cosh
from scipy.integrate import ode

"""Environ holds all the 'global' variables such as C, gK, gCa, etc.
gKS, Iext are the parameters that matter"""
class Environ(object):

    def __init__(self, gK = 9.0 , gCa=4.4, gL = 2.0, gKS=0.19, EK=-80., ECa=120., EL=-60., VCa=-1.2, kCa=2./18., VK=2., kK=2./10., kc=0.7, Vc=-25., C=2.4, Iext=37., de=0.052, ep=2.5):
        self.gK = gK
        self.gCa = gCa
        self.gL = gL
        self.gKS = gKS
        self.EK = EK
        self.ECa = ECa
        self.EL = EL
        self.VCa = VCa
        self.VK = VK
        self.kCa = kCa
        self.kK = kK
        self.kc = kc
        self.Vc = Vc
        self.C = C
        self.Iext = Iext
        self.ep = ep
        self.de = de
        self.V1 = VCa
        self.V2 = 2./kCa
        self.V3 = VK
        self.V4 = 2/kK
        #note: kK = 2/V4 as in the MATLAB code, VCa = V1, kCa = 2/V2, V3 = VK, Vc kc = 2*kc equal
"""Model represents the model in Ghiglia, Holmes paper:
Cv' = - [ICa + IK + IL + IKS] + Iext
m' = ep/tm(v)*(mInf(v)-m)
c' = del/tc(v)*(cInf(v)-c)

ICa = gCa*nInf(v)(v-ECa)
IL = gL*(v-EK)
IK = gK*m*(v-EK)
IKS = gKS*c*(v-EK)
wInf(v) = 1/(1+exp(-kw(v-vwth))) = 0.5*(1+tanh(-kw(v-vwth)/2)) (w = m,c,n)
tw(v) = sech(kw(v-vwth)) (w = m,c,n)"""
class Model(object):
    def __init__(self,env):
        self.env = env

    def nInf(self,V):
        return 0.5*(1+np.tanh((V-self.env.VCa)*self.env.kCa/2))

    def mInf(self,V):
        return 0.5*(1+np.tanh((V-self.env.VK)*self.env.kK))

    def tm(self,V):
        return 1./np.cosh((V-self.env.VK)*self.env.kK/4)

    def cInf(self,V):
        return 0.5*(1+np.tanh(self.env.kc/2*(V-self.env.Vc)))

    def tc(self,V):
        return 1./np.cosh((V-self.env.Vc)*self.env.kK/4)
    
    def dyn(self,t,x):
        [V, m, c] = x
        mInf = 0.5*(1+tanh((V-self.env.V3)/self.env.V4))
        cInf = 0.5*(1+tanh((V-self.env.Vc)*self.env.kc))
        nInf = 0.5*(1+tanh((V-self.env.V1)/self.env.V2))
        #tm = 1./cosh((V-self.env.V3)/(2*self.env.V4))
        #tc = 1./cosh((V-self.env.Vc)*self.env.kK/4)
        ICa = self.env.gCa*nInf*(V-self.env.ECa)
        IK = self.env.gK*m*(V-self.env.EK)
        IKS = self.env.gKS*c*(V-self.env.EK)
        IL = self.env.gL*(V-self.env.EL) #EK in paper
        
        Vd = -1./self.env.C*(ICa + IK + IL + IKS) + self.env.Iext/self.env.C
        md = self.env.ep*(mInf-m)*cosh((V-self.env.V3)/(2*self.env.V4))
        cd = self.env.de*(cInf-c)*cosh((V-self.env.Vc)/(2*self.env.V4))
        
        return np.array([Vd, md, cd])


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
        r.set_initial_value(np.array(x0),t0)
        
        t = [t0]
        xd = [np.array(x0)]

        while r.successful() and r.t < tmax:
            r.integrate(r.t + dt)
            t.append(r.t)
            xd.append(r.y)
#            print("%g %g %g %g" %(r.t, r.y[0],r.y[1],r.y[2]))

        print(r.successful())
        return [t, xd]