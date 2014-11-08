import scipy as sp
import pylab as plt
import numpy as np
from scipy.integrate import odeint
from ndtools.spikes import *
matplotlib.pyplot.close("all")
########################################
########################################
#Hodgkin-Huxley Neurons definitions

C_m  =   1.0 # membrane capacitance, in uF/cm^2
g_K  =   36.0 # maximum conducances, in mS/cm^2
g_L  =   0.3
g_Na =   120
E_K  =  -82.0
E_L  =  -59.387
E_Na =   45.0

t_start = 0
t_stop = 300
t_step = 0.3
t = sp.arange(t_start, t_stop, t_step)

def alpha_m(V): return  0.1 *( V + 45 ) / ( 1 - sp.exp( -(V + 45)/10))
def beta_m(V) : return  4 * sp.exp(-(V + 70)/18)
def alpha_h(V): return  0.07 * sp.exp(-(V + 70)/20)
def beta_h(V) : return  1/(1 + sp.exp(-(V + 40)/10))
def alpha_n(V): return  0.01* (V + 60)/(1 - sp.exp(-(V + 60)/10))
def beta_n(V) : return  0.125 * sp.exp(-(V + 70)/80)
def I_K(V, n):
	return g_K   * n**4          * (V - E_K)
def I_Na(V, m, h):
	return g_Na  * m**3  *h      * (V - E_Na)
def I_L(V): return g_L * (V - E_L)
def deriv_V_t(V, m, h, n, I):
    return (1. / C_m) * (- I_Na(V, m, h) - I_K(V, n) - I_L(V) + I)
def dmdt(V,m):
	return alpha_m(V)*(1-m)-beta_m(V)*m
def dhdt(V,h):
	return alpha_h(V)*(1-h)-beta_h(V)*h
def dndt(V,n):
	return alpha_n(V)*(1-n)-beta_n(V)*n
#################################
##uncoupled
#def dALLdt(X,t):
#	V0,V1, m0,m1, h0,h1, n0,n1= X
#	return\
#	deriv_V_t(V0, m0, h0, n0, I_ext[0]),\
#	deriv_V_t(V1, m1, h1, n1, I_ext[1]),\
#	dmdt(V0,m0),	dmdt(V1,m1),\
#	dhdt(V0,h0),	dhdt(V1,h1),\
#	dndt(V0,n0),	dndt(V1,n1)
#STORE= odeint(dALLdt, ( -70,-70,  0.3,0.3,  0.05,0.05,  0.6,0.6), t,)
#V = [STORE[:,0],STORE[:,1]] # the first  column is the V values
#m = [STORE[:,2],STORE[:,3]] # the second column is the m values
#h = [STORE[:,4],STORE[:,5]] # the second column is the h values
#n = [STORE[:,6],STORE[:,7]] # the second column is the n values
#plt.figure()
#plot(t, V[0], label="$V_{A}$")
#plot(t, V[1], label="$V_{B}$")
#plt.ylabel('$Membrane Voltage$ ')
#plt.legend(loc="upper right")
#plt.xlabel('t ($ms$)')
#############################
# inhibitory synaptic current
#g_GABA=0#sp.arange(0.5, 3.5, t_step)
E_Cl   =-80
alpha_r=5
#beta_r =0.18
Tmax   =1.5
Kp     =5
Vp     =7

def I_syn_inhi(r, V_post):
	return g_GABA*r*(V_post-E_Cl)
def deriv_V_t_inhi(V, m, h, n, r, I_ext):
    return (1. / C_m) * (-I_Na(V,m,h) -I_K(V,n) -I_L(V) -I_syn_inhi(r,V)+ I_ext)
def drdt_inhi(T,r):
	return alpha_r * T * ( 1-r ) - beta_r * r
def Tconc_inhi(V_pre):
	return Tmax/(1+ sp.exp(-(V_pre-Vp)/Kp))
#def single(V,Vpre, m, h, n, r, I_ext, t):
#	T=Tconc_inhi(Vpre)
#	dV=deriv_V_t_inhi(V,m,h,n,r,I_ext)
#	dm=dmdt(V,m)
#	dh=dhdt(V,h)
#	dn=dndt(V,n)
#	dr=drdt_inhi(T,r)
#	return dV, dm, dh, dn, dr
#def dALLdt(X,t):
#	V0,V1, m0,m1, h0,h1, n0,n1, r0,r1= X
#	#V_post, V_pre, m_post, m_pre......
#	N_A=single(V0, V1, m0, h0, n0, r0, 10,t)
#	N_B=single(V1, V0, m1, h1, n1, r1, 20,t)
#	return\
#	N_A[0],N_B[0],\
#	N_A[1],N_B[1],\
#	N_A[2],N_B[2],\
#	N_A[3],N_B[3],\
#	N_A[4],N_B[4]
def dALLdt(X,t):
	V0,V1, m0,m1, h0,h1, n0,n1, r0,r1= X
	T0=Tconc_inhi(V1) #Vpre for synapse A is V_B
	T1=Tconc_inhi(V0) #Vpre for synapse B is V_A
	return\
	deriv_V_t_inhi(V0, m0, h0, n0, r0,  I_ext[0]),\
	deriv_V_t_inhi(V1, m1, h1, n1, r1,  I_ext[1]),\
	dmdt(V0,m0),	dmdt(V1,m1),\
	dhdt(V0,h0),	dhdt(V1,h1),\
	dndt(V0,n0),	dndt(V1,n1),\
	drdt_inhi(T0,r0),	drdt_inhi(T1,r1)
#################################
#1.2 constant gGABA
#I_ext=[10,20]
#STORE= odeint(dALLdt, ( -70,-70,  m_0,m_0,  h_0,h_0,  n_0,n_0, 0,0), t,)
#V = [STORE[:,0],STORE[:,1]] # the first  column is the V values
#m = [STORE[:,2],STORE[:,3]] # the second column is the m values
#h = [STORE[:,4],STORE[:,5]] # the second column is the h values
#n = [STORE[:,6],STORE[:,7]] # the second column is the n values
#r = [STORE[:,8],STORE[:,9]] # the second column is the n values
#plt.figure()
#plot(t, V[0], label="$V_{A}$")
#plot(t, V[1], label="$V_{B}$")
#plt.ylabel('$Membrane Voltage$ ')
#plt.legend(loc="upper right")
#plt.xlabel('t ($ms$)')
#print(isi(sp.arange(t_start, t_stop, t_step),V[0],0 ))

############################
##1.2-variable gGABA
#I_ext=[10,20]
#g_GABAs_step = sp.arange(0, 0.8, 0.2)
#V_ABcoupling=zeros((len(t),2*len(g_GABAs_step)))
#for idx, G_GABAtest in enumerate(g_GABAs_step):
#	g_GABA=G_GABAtest
#	STORE= odeint(dALLdt, ( -70,-70,  m_0,m_0,  h_0,h_0,  n_0,n_0, 0,0), t,)
#	V_ABcoupling[:,idx                  ]  = STORE[:,0]# V_ABcoupling[0,1,2,3,4] stores V_A
#	V_ABcoupling[:,idx+len(g_GABAs_step)]  = STORE[:,1] # V_ABcoupling[4,5,6,7,8] stores V_B
#
#fig2, axes2 = plt.subplots(len(g_GABAs_step), 1, figsize=(6, 2 * len(g_GABAs_step)))
#for idx, ax in enumerate(axes2):
#	mean1, std1 =isi(sp.arange(t_start+100, t_stop, t_step),V_ABcoupling[:,idx][100/t_step:])
#	mean2, std2 =isi(sp.arange(t_start+100, t_stop, t_step),V_ABcoupling[:,idx+len(g_GABAs_step)][100/t_step:])
#	freq1=(1/mean1)
#	print(std1)
#	freq2=(1/mean2)
#	print(std2)
#	ax.plot(t, V_ABcoupling[:,idx])
#	ax.plot(t, V_ABcoupling[:,idx + len(g_GABAs_step)])
#	ax.set_title(r'$g_{GABA} =$ ' + str(g_GABAs_step[idx])+" $Spiking Freq= $"+ str(freq1)+"$,$ "+str(freq2))
#fig2.tight_layout()
##################################
##1.3 In-phase oscillations
#beta_r_step =sp.arange(0.5, 0, -0.1)
#V_ABcoupling=zeros((len(t),2*len(beta_r_step)))
#I_ext=[10,10.1]
#g_GABA=1
#for idx, beta_r_test in enumerate(beta_r_step):
#	beta_r=beta_r_test
#	STORE= odeint(dALLdt, ( -70,-70,  m_0,m_0,  h_0,h_0,  n_0,n_0, 0,0), t,)
#	V_ABcoupling[:,idx                  ]  = STORE[:,0]# V_ABcoupling[0,1,2,3,4] stores V_A
#	V_ABcoupling[:,idx+len(beta_r_step)]   = STORE[:,1] # V_ABcoupling[4,5,6,7,8] stores V_B
#
#fig2, axes2 = plt.subplots(len(beta_r_step), 1, figsize=(6, 2 * len(beta_r_step)))
#for idx, ax in enumerate(axes2):
#	mean1, std1 =isi(sp.arange(t_start+100, t_stop, t_step),V_ABcoupling[:,idx][100/t_step:],-15)
#	mean2, std2 =isi(sp.arange(t_start+100, t_stop, t_step),V_ABcoupling[:,idx+len(beta_r_step)][100/t_step:],-15)
#	freq1=(1/mean1)
#	print(std1)
#	freq2=(1/mean2)
#	print(std2)
#	ax.plot(t, V_ABcoupling[:,idx])
#	ax.plot(t, V_ABcoupling[:,idx + len(beta_r_step)])
#	ax.set_title(r'$\beta_{r} =$ ' + str(beta_r_step[idx])+" $Spiking Freq= $"+ str(freq1)+"$,$ "+str(freq2))
#fig2.tight_layout()


####################################################################
## excitary synapse
E_exci=-38
alpha_r_exci=2.4
beta_r_exci=0.56
Tmax_exci=1
#g_Glu=0

def I_syn_exci(r, V_post):
	return g_Glu*r*(V_post-E_exci)

def Tconc_exci(V_pre):
	return Tmax_exci/(1+ sp.exp(-(V_pre-Vp)/Kp))

#def deriv_V_t_exci(V, m, h, n, r, I_ext):
#    return (1. / C_m) * (-I_Na(V,m,h) -I_K(V,n) -I_L(V) -I_syn_exci(r,V)+ I_ext)

def drdt_exci(T,r):
	return alpha_r_exci * T * ( 1-r ) - beta_r_exci * r

####2.1 excitary synapse model
#def A_excite_B(X,t):
#	#Va, Vb......
#	V0,V1, m0,m1, h0,h1, n0,n1, r1= X
#	T1=Tconc_exci(V0) #Vpre for synapse B is V_A
#	return\
#	deriv_V_t      (V0, m0, h0, n0,    I_ext[0]),\
#	deriv_V_t      (V1, m1, h1, n1,  ( I_ext[1]-I_syn_exci(r1,V1) ) ),\
#	dmdt(V0,m0),	dmdt(V1,m1),\
#	dhdt(V0,h0),	dhdt(V1,h1),\
#	dndt(V0,n0),	dndt(V1,n1),\
#	drdt_exci(T1,r1)# drdt for t0 is a dummy variable as it doesn't exist
#I_ext=[10,0]
#g_Glu_step = sp.arange(0.3, 0.6, 0.1)
#V_ABcoupling=zeros((len(t),2*len(g_Glu_step)))
#for idx, g_Glu_test in enumerate(g_Glu_step):
#	g_Glu=g_Glu_test
#	STORE= odeint(A_excite_B, ( -70,-70,  m_0,m_0,  h_0,h_0,  n_0,n_0,0), t,)
#	V_ABcoupling[:,idx                  ]  = STORE[:,0]# V_ABcoupling[0,1,2,3,4] stores V_A
#	V_ABcoupling[:,idx+len(g_Glu_step)]  = STORE[:,1] # V_ABcoupling[4,5,6,7,8] stores V_B
#fig2, axes2 = plt.subplots(len(g_Glu_step), 1, figsize=(6, 2 * len(g_Glu_step)))
#for idx, ax in enumerate(axes2):
#	mean1, std1 =isi(sp.arange(t_start+100, t_stop, t_step),V_ABcoupling[:,idx][100/t_step:])
#	mean2, std2 =isi(sp.arange(t_start+100, t_stop, t_step),V_ABcoupling[:,idx+len(g_Glu_step)][100/t_step:])
#	freq1=(1/mean1)
#	print(std1)
#	freq2=(1/mean2)
#	print(std2)
#	ax.plot(t, V_ABcoupling[:,idx])
#	ax.plot(t, V_ABcoupling[:,idx + len(g_Glu_step)])
#	ax.set_title(r'$g_{Glu} =$ ' + str(g_Glu_step[idx])+" $Spiking Freq= $"+ str(freq1)+"$,$ "+str(freq2))
#fig2.tight_layout()
#
######################################################
####2.2 feedforward inhibition
#
#def FFIH(X,t):
#	#Va, Vb..................................0   0         0    exc  inhbi
#	Va,Vb,Vc, ma,mb,mc, ha,hb,hc, na,nb,nc,r_AexciB, r_AexciC, r_BinhiC= X
##	Ta     =0 #No synapse connecting A
#	T_AexciB=Tconc_exci(Va) #Vpre for synapse B is V_A
#	T_BinhiC=Tconc_inhi(Vb)
#	T_AexciC=Tconc_exci(Va)
#
#	return\
#	deriv_V_t(Va, ma, ha, na,  I_ext[0]                                            ),\
#	deriv_V_t(Vb, mb, hb, nb, (I_ext[1] - I_syn_exci(r_AexciB,Vb))                      ),\
#	deriv_V_t(Vc, mc, hc, nc, (I_ext[2] - I_syn_exci(r_AexciC,Vc) -I_syn_inhi(r_BinhiC, Vc)) ),\
#	dmdt(Va,ma),	dmdt(Vb,mb),	dmdt(Vc,mc),\
#	dhdt(Va,ha),	dhdt(Vb,hb),	dhdt(Vc,hc),\
#	dndt(Va,na),	dndt(Vb,nb),	dndt(Vc,nc),\
#	drdt_exci(T_AexciB,r_AexciB),\
#	drdt_exci(T_AexciC,r_AexciC), drdt_inhi(T_BinhiC, r_BinhiC)
#g_Glu  = 0.5
#beta_r = 0.1
#g_GABA = 2
#n_0 = 0.31773
#m_0 = 0.05296
#h_0 = 0.59599
#I_ext_step=[[10,0,0],[15,0,0],[20,0,0],[30,0,0]]
#V_ABCcoupling=zeros((len(t),3*len(I_ext_step)))
#for idx, i_ext_A in enumerate(I_ext_step):
#	I_ext=i_ext_A
#	STORE= odeint(FFIH,\
#	( -70,-70,-70,     \
#	  m_0,m_0,m_0,     \
#	  h_0,h_0,h_0,     \
#	  n_0,n_0,n_0,     \
#	  0,0,0), t,  )
#	V_ABCcoupling[:,idx                   ]  = STORE[:,0]# V_ABcoupling[0,1,2] stores V_A
#	V_ABCcoupling[:,idx+   len(I_ext_step)]  = STORE[:,1]# V_ABcoupling[3,4,5] stores V_B
#	V_ABCcoupling[:,idx+ 2*len(I_ext_step)]  = STORE[:,2]# V_ABcoupling[6,7,8] stores V_C
#fig2, axes2 = plt.subplots( len(I_ext_step), 1)#, figsize=(6, 3 * len(I_ext_step)))
#for idx, ax in enumerate(axes2):
##	mean1, std1 =isi(sp.arange(t_start+100, t_stop, t_step),V_ABcoupling[:,idx][100/t_step:])
##	mean2, std2 =isi(sp.arange(t_start+100, t_stop, t_step),V_ABcoupling[:,idx+len(I_ext_step)][100/t_step:])
##	freq1=(1/mean1)
##	print(std1)
##	freq2=(1/mean2)
##	print(std2)
#	ax.plot(t, V_ABCcoupling[:,idx                    ],label='$V_{A}$')
#	ax.plot(t, V_ABCcoupling[:,idx +   len(I_ext_step)],label='$V_{B}$')
#	ax.plot(t, V_ABCcoupling[:,idx + 2*len(I_ext_step)],label='$V_{C}$')
#	ax.legend(loc="upper right")
#	ax.set_title(r'$I_{ext} =$ '+ str(I_ext_step[idx])+'$, g_{GLU}=$'+str(g_Glu)+', $g_{GABA}=$'+str(g_GABA))#+" $Spiking Freq= $"+ str(freq1)+"$,$ "+str(freq2))
#fig2.tight_layout()

######################################################
####2.2 feedback inhibition
#
#def FBIH(X,t):
#	#Va, Vb..............................
#	Va,Vb,Vc, ma,mb,mc, ha,hb,hc, na,nb,nc,r_exi_A2C,r_exi_C2B, r_inh_B2C= X
##	Ta     =0 #No synapse connecting A
#	Tb_exci=Tconc_exci(Vc) #Vpre for synapse B is V_A
#	Tc_inhi=Tconc_inhi(Vb)
#	Tc_exci=Tconc_exci(Va)
#
#	return\
#	deriv_V_t(Va, ma, ha, na,  I_ext[0]                                            ),\
#	deriv_V_t(Vb, mb, hb, nb, (I_ext[1] - I_syn_exci(r_exi_C2B,Vb))                      ),\
#	deriv_V_t(Vc, mc, hc, nc, (I_ext[2] - I_syn_exci(r_exi_A2C,Vc) -I_syn_inhi(r_inh_B2C, Vc)) ),\
#	dmdt(Va,ma),	dmdt(Vb,mb),	dmdt(Vc,mc),\
#	dhdt(Va,ha),	dhdt(Vb,hb),	dhdt(Vc,hc),\
#	dndt(Va,na),	dndt(Vb,nb),	dndt(Vc,nc),\
#	drdt_exci(Tc_exci,r_exi_A2C),drdt_exci(Tb_exci,r_exi_C2B), drdt_inhi(Tc_inhi, r_inh_B2C)
#g_Glu  = 0.5
#beta_r = 0.1
#g_GABA = 2
#n_0 = 0.31773
#m_0 = 0.05296
#h_0 = 0.59599
#I_ext_step=[[10,0,0],[15,0,0],[20,0,0],[30,0,0]]
#V_ABCcoupling=zeros((len(t),3*len(I_ext_step)))
#for idx, i_ext_A in enumerate(I_ext_step):
#	I_ext=i_ext_A
#	STORE= odeint(FBIH,\
#	( -70,-70,-70,     \
#	  m_0,m_0,m_0,     \
#	  h_0,h_0,h_0,     \
#	  n_0,n_0,n_0,     \
#	  0,0,0), t,  )
#	V_ABCcoupling[:,idx                   ]  = STORE[:,0]# V_ABcoupling[0,1,2] stores V_A
#	V_ABCcoupling[:,idx+   len(I_ext_step)]  = STORE[:,1]# V_ABcoupling[3,4,5] stores V_B
#	V_ABCcoupling[:,idx+ 2*len(I_ext_step)]  = STORE[:,2]# V_ABcoupling[6,7,8] stores V_C
#fig2, axes2 = plt.subplots( len(I_ext_step), 1)#, figsize=(6, 3 * len(I_ext_step)))
#for idx, ax in enumerate(axes2):
##	mean1, std1 =isi(sp.arange(t_start+100, t_stop, t_step),V_ABcoupling[:,idx][100/t_step:])
##	mean2, std2 =isi(sp.arange(t_start+100, t_stop, t_step),V_ABcoupling[:,idx+len(I_ext_step)][100/t_step:])
##	freq1=(1/mean1)
##	print(std1)
##	freq2=(1/mean2)
##	print(std2)
#	ax.plot(t, V_ABCcoupling[:,idx                    ],label='$V_{A}$')
#	ax.plot(t, V_ABCcoupling[:,idx +   len(I_ext_step)],label='$V_{B}$')
#	ax.plot(t, V_ABCcoupling[:,idx + 2*len(I_ext_step)],label='$V_{C}$')
#	ax.legend(loc="upper right")
#	ax.set_title(r'$I_{ext} =$ '+ str(I_ext_step[idx])+'$, g_{GLU}=$'+str(g_Glu)+', $g_{GABA}=$'+str(g_GABA))#+" $Spiking Freq= $"+ str(freq1)+"$,$ "+str(freq2))
#fig2.tight_layout()

##########################################
##2.5 BONUS loop

##########################################
##2.5 BONUS loop
#def LOOP3(X,t):
#	#Va, Vb..................................0   0         0    exc  inhbi
#	Va,Vb,Vc, ma,mb,mc, ha,hb,hc, na,nb,nc,r_CexciA, r_AexciB, r_BexciC= X
##	Ta     =0 #No synapse connecting A
#	T_CexciA=Tconc_exci(Vc)
#	T_AexciB=Tconc_exci(Va) #Vpre for synapse B is V_A
#	T_BexciC=Tconc_inhi(Vb)
#	return\
#	deriv_V_t(Va, ma, ha, na, (I_ext[0]*(t <= 10) - I_syn_exci(r_CexciA,Va))                                         ),\
#	deriv_V_t(Vb, mb, hb, nb, (I_ext[1] - I_syn_exci(r_AexciB,Vb))                      ),\
#	deriv_V_t(Vc, mc, hc, nc, (I_ext[2] - I_syn_exci(r_BexciC,Vc)) ),\
#	dmdt(Va,ma),	dmdt(Vb,mb),	dmdt(Vc,mc),\
#	dhdt(Va,ha),	dhdt(Vb,hb),	dhdt(Vc,hc),\
#	dndt(Va,na),	dndt(Vb,nb),	dndt(Vc,nc),\
#	drdt_exci(T_CexciA,r_CexciA),drdt_exci(T_AexciB,r_AexciB), drdt_exci(T_BexciC, r_BexciC)
#g_Glu  = 0.3
#n_0 = 0.31773
#m_0 = 0.05296
#h_0 = 0.59599
#I_ext_step=[[10,0,0],[15,0,0]]
#V_ABCcoupling=zeros((len(t),3*len(I_ext_step)))
#for idx, i_ext_A in enumerate(I_ext_step):
#	I_ext=i_ext_A
#	STORE= odeint(LOOP3,\
#	( -70,-70,-70,     \
#	  m_0,m_0,m_0,     \
#	  h_0,h_0,h_0,     \
#	  n_0,n_0,n_0,     \
#	  0,0,0), t,  )
#	V_ABCcoupling[:,idx                   ]  = STORE[:,0]# V_ABcoupling[0,1,2] stores V_A
#	V_ABCcoupling[:,idx+   len(I_ext_step)]  = STORE[:,1]# V_ABcoupling[3,4,5] stores V_B
#	V_ABCcoupling[:,idx+ 2*len(I_ext_step)]  = STORE[:,2]# V_ABcoupling[6,7,8] stores V_C
#fig2, axes2 = plt.subplots( len(I_ext_step), 1)#, figsize=(6, 3 * len(I_ext_step)))
#for idx, ax in enumerate(axes2):
##	mean1, std1 =isi(sp.arange(t_start+100, t_stop, t_step),V_ABcoupling[:,idx][100/t_step:])
##	mean2, std2 =isi(sp.arange(t_start+100, t_stop, t_step),V_ABcoupling[:,idx+len(I_ext_step)][100/t_step:])
##	freq1=(1/mean1)
##	print(std1)
##	freq2=(1/mean2)
##	print(std2)
#	ax.plot(t, V_ABCcoupling[:,idx                    ],label='$V_{A}$')
#	ax.plot(t, V_ABCcoupling[:,idx +   len(I_ext_step)],label='$V_{B}$')
#	ax.plot(t, V_ABCcoupling[:,idx + 2*len(I_ext_step)],label='$V_{C}$')
#	ax.legend(loc="upper right")
#	ax.set_title(r'$I_{ext} =$ '+ str(I_ext_step[idx])+'$, g_{GLU}=$'+str(g_Glu)+', $g_{GABA}=$'+str(g_GABA))#+" $Spiking Freq= $"+ str(freq1)+"$,$ "+str(freq2))
#fig2.tight_layout()

#################################################
#def LOOP5(X,t):
#	#Va, Vb..................................0   0         0    exc  inhbi
#	Va,Vb,Vc,Vd,Ve, \
#	ma,mb,mc,md,me, \
#	ha,hb,hc,hd,he, \
#	na,nb,nc,nd,ne, \
#	r_EexciA, r_AexciB, r_BexciC,r_CexciD,r_DexciE= X
##	Ta     =0 #No synapse connecting A
#	T_EexciA=Tconc_exci(Ve)
#	T_AexciB=Tconc_exci(Va) #Vpre for synapse B is V_A
#	T_BexciC=Tconc_inhi(Vb)
#	T_CexciD=Tconc_inhi(Vc)
#	T_DexciE=Tconc_inhi(Vd)
#	return\
#	deriv_V_t(Va, ma, ha, na, (I_ext[0]*(t <= 1)- I_syn_exci(r_EexciA,Va))  ),\
#	deriv_V_t(Vb, mb, hb, nb, (I_ext[1]         - I_syn_exci(r_AexciB,Vb))  ),\
#	deriv_V_t(Vc, mc, hc, nc, (I_ext[2]         - I_syn_exci(r_BexciC,Vc))  ),\
#	deriv_V_t(Vd, md, hd, nd, (I_ext[3]         - I_syn_exci(r_CexciD,Vd))  ),\
#	deriv_V_t(Ve, me, he, ne, (I_ext[4]         - I_syn_exci(r_DexciE,Ve))  ),\
#	dmdt(Va,ma),	dmdt(Vb,mb),	dmdt(Vc,mc), dmdt(Vd,md), dmdt(Ve,me),\
#	dhdt(Va,ha),	dhdt(Vb,hb),	dhdt(Vc,hc), dhdt(Vd,hd), dhdt(Ve,he),\
#	dndt(Va,na),	dndt(Vb,nb),	dndt(Vc,nc), dndt(Vd,nd), dndt(Ve,ne),\
#	drdt_exci(T_EexciA,r_EexciA),\
#	drdt_exci(T_AexciB,r_AexciB),\
#	drdt_exci(T_BexciC,r_BexciC),\
#	drdt_exci(T_CexciD,r_CexciD),\
#	drdt_exci(T_DexciE,r_DexciE)
#g_Glu  = 0.3
#E_exci=-38
#n_0 = 0.31773
#m_0 = 0.05296
#h_0 = 0.59599
#I_ext_step=[[10,0,0,0,0],[15,0,0,0,0]]
#V_ABCcoupling=zeros((len(t),5*len(I_ext_step)))
#for idx, i_ext_A in enumerate(I_ext_step):
#	I_ext=i_ext_A
#	STORE= odeint(LOOP5,\
#	( -70,-70,-70,-70,-70,   \
#	  m_0,m_0,m_0,m_0,m_0,  \
#	  h_0,h_0,h_0,h_0,h_0,  \
#	  n_0,n_0,n_0,n_0,n_0,  \
#	  0,0,0,0,0), t,  )
#	V_ABCcoupling[:,idx                   ]  = STORE[:,0]# V_ABcoupling[0,1,2] stores V_A
#	V_ABCcoupling[:,idx+   len(I_ext_step)]  = STORE[:,1]# V_ABcoupling[3,4,5] stores V_B
#	V_ABCcoupling[:,idx+ 2*len(I_ext_step)]  = STORE[:,2]# V_ABcoupling[6,7,8] stores V_C
#	V_ABCcoupling[:,idx+ 3*len(I_ext_step)]  = STORE[:,3]# V_ABcoupling[3,4,5] stores V_D
#	V_ABCcoupling[:,idx+ 4*len(I_ext_step)]  = STORE[:,4]# V_ABcoupling[6,7,8] stores V_E
#fig2, axes2 = plt.subplots( len(I_ext_step), 1)#, figsize=(6, 3 * len(I_ext_step)))
#for idx, ax in enumerate(axes2):
##	mean1, std1 =isi(sp.arange(t_start+100, t_stop, t_step),V_ABcoupling[:,idx][100/t_step:])
##	mean2, std2 =isi(sp.arange(t_start+100, t_stop, t_step),V_ABcoupling[:,idx+len(I_ext_step)][100/t_step:])
##	freq1=(1/mean1)
##	print(std1)
##	freq2=(1/mean2)
##	print(std2)
#	ax.plot(t, V_ABCcoupling[:,idx                    ],label='$V_{A}$')
#	ax.plot(t, V_ABCcoupling[:,idx +   len(I_ext_step)],label='$V_{B}$')
#	ax.plot(t, V_ABCcoupling[:,idx + 2*len(I_ext_step)],label='$V_{C}$')
#	ax.plot(t, V_ABCcoupling[:,idx + 3*len(I_ext_step)],label='$V_{D}$')
#	ax.plot(t, V_ABCcoupling[:,idx + 4*len(I_ext_step)],label='$V_{E}$')
#	ax.legend(loc="upper right")
#	ax.set_title(r'$I_{ext} =$ '+ str(I_ext_step[idx])+'$, g_{GLU}=$'+str(g_Glu)+', $g_{GABA}=$'+str(g_GABA))#+" $Spiking Freq= $"+ str(freq1)+"$,$ "+str(freq2))
#fig2.tight_layout()
