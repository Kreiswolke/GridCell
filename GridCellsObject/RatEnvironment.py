__author__ = 'Oliver'
import numpy as np
import matplotlib.pyplot as plt
import random

# 1a)

sidelength = 1.25
dx = 0.1
dy = 0.1
#Creating Mesh of Rat Envrionment

xcoord = np.arange(-sidelength / 2, sidelength / 2, dx)
ycoord = np.arange(-sidelength / 2, sidelength / 2, dy)

xx, yy = np.meshgrid(xcoord, ycoord)

#Start time evolution in seconds
t_max = 50
t_0 = 0
#every time step is 10ms
dt = 0.1

speed = 0.4 * dt

Pos = np.zeros((t_max / dt, 2))
Pos[0] = [0.0, 0.0]
i = 0
angle = 0.0
for t in np.arange(t_0, t_max, dt):

    angle = np.random.normal(angle, 0.2)
    # print [np.sin(angle),np.cos(angle)]
    Pos[i] = Pos[i - 1] + ([np.sin(angle), np.cos(angle)] / (np.linalg.norm([np.sin(angle), np.cos(angle)]))) * speed

    while (abs(Pos[i, 0]) > sidelength / 2.0) or (abs(Pos[i, 1]) > sidelength / 2.0):
        angle = np.random.normal(angle, 0.2)
        Pos[i] = Pos[i - 1] + (
                                  [np.sin(angle), np.cos(angle)] / (np.linalg.norm([np.sin(angle),
                                                                                    np.cos(angle)]))) * speed

    i += 1

plt.plot(Pos[:, 0], Pos[:, 1])
plt.show()

print 'Number of steps:', t_max / dt



#1b)
#200 input neurons, which are randomly placed over the area
'''
def firingrate(RatLoc, PrefLoc, sig_p=0.05):
    rates = np.zeros(np.shape(RatLoc)[0])
    for i in np.arange(np.shape(RatLoc)[0]):
        rates[i] = np.exp(-((np.linalg.norm(RatLoc[i] - PrefLoc[i])) ** 2) / (2 * (sig_p ** 2)))
    return rates
'''
def firingrate(RatLoc, PrefLoc, sig_p=0.05):
    rates = np.exp(-((np.linalg.norm(RatLoc - PrefLoc)) ** 2) / (2 * (sig_p ** 2)))
    return rates


N_in = 200
sidelength = 1.25
x_pref = np.zeros((N_in, 2))

for i in np.arange(N_in):
    x_pref[i, 0] = random.uniform(-sidelength / 2, sidelength / 2)
    x_pref[i, 1] = random.uniform(-sidelength / 2, sidelength / 2)

rates=np.zeros((N_in,t_max/dt))

for i in np.arange(t_max/dt):
    for j in np.arange(N_in):
        rates[j,i] = firingrate(Pos[i], x_pref[j])

print rates
print np.shape(rates)
#print rates
plt.plot(x_pref[:, 0], x_pref[:, 1],'o')
plt.title('Preferred firing locations')
plt.show()



#1.4)
def generic_euler(f_func, ratesin, params, x_0, t_0, t_max, dt):
    t = np.arange(t_0, t_max, dt)
    columns = len(t)
    x = np.zeros((3, columns))
    x[:, 0] = x_0
    for i in range(1, columns):
        x[0, i] = x[0, i - 1] + (f_func[0](t[i - 1], ratesin[:,i], **params)) * dt
        x[1, i] = x[1, i - 1] + (f_func[1](x[0, i - 1], x[1, i - 1], t[i - 1], **params)) * dt
        x[2, i] = x[2, i - 1] + (f_func[2](x[0, i - 1], x[1, i - 1], x[2, i - 1], t[i - 1], **params)) * dt
    return t, x


def hfunc(t, rates, **params):
    return np.sin(np.pi * k * (t ** 2))


def rminusdot(h, rminus, t, **params):
    return (h - rminus) / tauminus


def rplusdot(h, rminus, rplus, t, **params):
    return (h - rplus - rminus) / tauplus


params = {'k': 0.1, 'tauminus': 0.3, 'tauplus': 0.1, 'h': 0.0, 'rminus': 0.0, 'rplus': 0.0}
initial = [0.0, 0.0, 0.0]
k = 0.1
#tauminus = 0.3
#tauplus = 0.1
'''
T, X = generic_euler([hfunc, rminusdot, rplusdot], rates, params, initial, t_0, t_max, dt)

plt.figure()

plt.subplot(2, 1, 1)
plt.plot(k * T, X[0, :])
plt.xlabel('frequency $k\cdot t$ in Hz')
plt.ylabel('input $h(t)$')

plt.subplot(2, 1, 2)
plt.plot(k * T, X[2, :])
plt.xlabel('frequency $k\cdot t$ in Hz')
plt.ylabel('activation variable $r^+$ in s')
plt.show()

'''
# 1.5)

# create random weights and normalize them to norm = 1

def randweights(inputneurons, eta=0.1):
    weights = np.zeros(inputneurons)
    for i in range(inputneurons):
        weights[i] = (1 - eta) + eta * (random.uniform(0, 1))
    weights = weights / (np.linalg.norm(weights))
    return weights


#print  randweights(N_in)

#Computing feed forward input

def h_i(t, ratesin, **params):
    #  Output=np.zeros(N_out)
    # for i in range(N_out):
    return np.inner(randweights(N_in), ratesin)


#neweuler
def generic_euler2(f_func, ratesin, params, x_0, t_0, t_max, dt):
    t = np.arange(t_0, t_max, dt)
    columns = len(t)
    x = np.zeros((2, columns))
    x[:, 0] = x_0
    for i in range(1, columns):
        x[0, i] = x[0, i - 1] + (f_func[0](ratesin[i-1], x[0, i - 1], t[i - 1], **params)) * dt
        x[1, i] = x[1, i - 1] + (f_func[1](ratesin[i-1], x[1, i - 1], x[1, i - 1], t[i - 1], **params)) * dt
    return  x

#print 'h_i', np.inner(randweights(N_in), rates)
N_out = 100
# 1.6)
'''
hinput=np.zeros((N_out,t_max/dt))
T=[]

for i in range(int(t_max/dt)):
    for j in range(N_out):
        hinput[j,i]=np.inner(randweights(N_in), rates[:,j])
    T.append(i)
#print 'blub', hinput

plt.plot(T,hinput[2,:])
plt.show()


#Rplus
rPLUS=np.zeros((N_out,t_max/dt))
for i in range(N_out):
    rPLUS[i,:]=generic_euler2([rminusdot, rplusdot], hinput[i,:], params, [1.0,0.0], t_0, t_max, dt)[1,:]

print 'RpLUS', rPLUS

#T5, rplus = generic_euler([rminusdot, rplusdot], rates, params, initial, t_0, t_max, dt)



def heaviside(the_x):
    if the_x > 0:
        the_result = 1
    elif the_x == 0:
        the_result = 0.5
    else:
        the_result = 0

    return the_result

def eulerstep(f_func, startvalue, params, dt):
    return startvalue + f_func(startvalue, params) * dt

def h_i2(ratesin):
    #  Output=np.zeros(N_out)
    # for i in range(N_out):
    return np.inner(randweights(N_in), ratesin)

def rminusdot2(rminus, params):
    h=params['h']
    tauminus=params['tauminus']

    return (h - rminus) / tauminus


def rplusdot2( rplus, params):
    h=params['h']
    rminus=params['rminus']
    tauplus=params['tauplus']

    return (h - rplus - rminus) / tauplus

def OutputRate2( gain, threshold, params):
    rplus=params['rplus']
    rminus=params['rminus']

    rminus=eulerstep(rminusdot2, rminus, params, dt)
    params['rminus']=rminus

    rplus=eulerstep(rplusdot2, rplus, params, dt)
    params['rplus']=rplus


    return rplus, (2 / np.pi) * np.arctan(gain * (rplus - threshold)) * heaviside(rplus - threshold)
# We compute the Output rate for ever time step. For every time step there are iterations for every output neuron


def OutputRate(r_plus, gain, threshold):
    return (2 / np.pi) * np.arctan(gain * (r_plus - threshold)) * heaviside(r_plus - threshold)


#print OutputRate(N_out,[0.5,0.5],rplus[5])
A=np.zeros(t_max/dt)
for t in np.arange(t_0,t_max,dt):
    i=0
    rplus=params['rplus']
    rminus=params['rminus']

    rminus=eulerstep(rminusdot2, rminus, params, dt)
    params['rminus']=rminus

    rplus=eulerstep(rplusdot2, rplus, params, dt)
    params['rplus']=rplus

    A[i]=rplus
    i+=1

plt.plot(np.arange(t_0,t_max,dt),A)
plt.show()

'''
# Outer time Loop
# initial values for gain a and threshold s
g=1.0
s=0.0
N_out=100.0
r_out=np.zeros((N_out,t_max/dt))
for t in np.arange(t_0+dt,t_max,dt):
    # Inner iteration Loop over Output neurons
    #r_out=np.zeros(N_o:ut)

    for i in range(int(N_out)):
        params['h']=h_i2(rates[:,t/dt])

        bla, r_out[i,t/dt]=OutputRate2(g, s, params)
        print 'rplus', bla
        print r_out[:,t/dt]
        #r_out[i,t/dt]=OutputRate(rPLUS[i,t/dt], g, s)
        #print 'rplus', rPLUS[i,t/dt]
        #print 'rout', r_out[i,t/dt]


    g = np.sum(r_out[:,t/dt]) / N_out
    s = ((np.sum(r_out[:,t/dt])) ** 2) / (N_out * (np.sum(r_out[:,t/dt]) ** 2))

    print g, s

    if (g > 0.11) or (g < 0.09):
        while (g > 0.11) or (g < 0.09):
            g=g+0.01*(g-0.1)
    if  (s > 0.33) or (s < 0.27):
        while (s > 0.33) or (s < 0.27):
            s=s+0.1*s*(s-0.3)

print r_out

#plt.plot(np.arange(t_0, t_max, dt),rates[5,:],'r')
#plt.plot(np.arange(t_0, t_max, dt),r_out[6,:])
#plt.show()


