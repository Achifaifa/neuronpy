import numpy as np
from scipy.integrate import ode
import warnings
"""Environ holds all the 'global' variables such as C, gK, gCa, etc.
gKS, Iext are the parameters that matter"""
class Environ(object):

    def __init__(self, gK = 9.0 , gCa=4.4, gL = 2.0, gKS=0.25, EK=-80., ECa=120., EL=-60., VCa=-1.2, kCa=0.11, VK=2., kK=0.2, kc=0.8, Vc=-27., C=1.2, Iext=35.6, de=0.052, ep=4.9):
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
        return 0.5*(1+np.tanh((V-self.env.VK)*self.env.kK/2))

    def tm(self,V):
        return 1./np.cosh((V-self.env.VK)*self.env.kK)

    def cInf(self,V):
        return 0.5*(1+np.tanh(self.env.kc/2*(V-self.env.Vc)))

    def tc(self,V):
        return 1./np.cosh((V-self.env.Vc)*self.env.kc)

    def dyn(self,t,x):
        [V, m, c] = x
        mInf = 0.5*(1+np.tanh((V-self.env.VK)*self.env.kK/2))
        cInf = 0.5*(1+np.tanh(self.env.kc/2*(V-self.env.Vc)))
        nInf = 0.5*(1+np.tanh((V-self.env.VCa)*self.env.kCa/2))
        tm = 1./np.cosh((V-self.env.VK)*self.env.kK)
        tc = 1./np.cosh((V-self.env.Vc)*self.env.kc)
        ICa = self.env.gCa*nInf*(V-self.env.ECa)
        IK = self.env.gK*m*(V-self.env.EK)
        IKS = self.env.gKS*c*(V-self.env.EK)
        IL = self.env.gL*(V-self.env.EL) #EK in paper
        
        Vd = -1./self.env.C*(ICa + IK + IL + IKS) + self.env.Iext/self.env.C
        md = self.env.ep*(mInf-m)/tm
        cd = self.env.de*(cInf-c)/tc
        
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
        r = ode(self.dyn).set_integrator('vode',nsteps=1500)
        r.set_initial_value(x0,t0)
        #r._integrator.iwork[2] = -1
        
        t = []
        xd = []
       # warnings.filterwarnings("ignore",category=UserWarning)
        while r.successful() and r.t < tmax:
            r.integrate(tmax, step = True)
            t.append(r.t)
            xd.append(r.y)
#            print("%g %g %g %g" %(r.t, r.y[0],r.y[1],r.y[2]))
     #   warnings.resetwarnings()
        print(r.successful())
        return [t, xd]