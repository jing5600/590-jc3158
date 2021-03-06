import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from   scipy.optimize import minimize
from sklearn.model_selection import train_test_split

import os

os.getcwd()

#USER PARAMETERS
IPLOT=True
INPUT_FILE='weight.json'
FILE_TYPE="json"
DATA_KEYS=['x','is_adult','y']
OPT_ALGO='BFGS'

#UNCOMMENT FOR VARIOUS MODEL CHOICES (ONE AT A TIME)
model_type="logistic"; NFIT=4; xcol=1; ycol=2;
# model_type="linear";   NFIT=2; xcol=1; ycol=2; 
# model_type="logistic";   NFIT=4; xcol=2; ycol=0;

#READ FILE
with open(INPUT_FILE) as f:
	my_input = json.load(f)  #read into dictionary

#CONVERT INPUT INTO ONE LARGE MATRIX (SIMILAR TO PANDAS DF)
X=[];
for key in my_input.keys():
	if(key in DATA_KEYS): X.append(my_input[key])

#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
X=np.transpose(np.array(X))

#SELECT COLUMNS FOR TRAINING 
x=X[:,xcol];  y=X[:,ycol]

#EXTRACT AGE<18
if(model_type=="linear"):
	y=y[x[:]<18]; x=x[x[:]<18]; 

#COMPUTE BEFORE PARTITION AND SAVE FOR LATER
XMEAN=np.mean(x); XSTD=np.std(x)
YMEAN=np.mean(y); YSTD=np.std(y)

#NORMALIZE
x=(x-XMEAN)/XSTD;  y=(y-YMEAN)/YSTD; 

#PARTITION
f_train=0.8; f_val=0.2
rand_indices = np.random.permutation(x.shape[0])
CUT1=int(f_train*x.shape[0]); 
train_idx,  val_idx = rand_indices[:CUT1], rand_indices[CUT1:]
xt=x[train_idx]; yt=y[train_idx]; xv=x[val_idx];   yv=y[val_idx]

#MODEL
def model(x,p):
	if(model_type=="linear"):   return  p[0]*x+p[1]  
	if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.01))))

#SAVE HISTORY FOR PLOTTING AT THE END
iteration=0; iterations=[]; loss_train=[];  loss_val=[]

#LOSS FUNCTION
def loss(p,xt,yt):
	global iterations,loss_train,loss_val,iteration

	#TRAINING LOSS
	yp=model(xt,p) #model predictions for given parameterization p
	training_loss=(np.mean((yp-yt)**2.0))  #MSE

	#VALIDATION LOSS
	yp=model(xv,p) #model predictions for given parameterization p
	validation_loss=(np.mean((yp-yv)**2.0))  #MSE

	#WRITE TO SCREEN
	if(iteration==0):    print("iteration	training_loss	validation_loss") 
	if(iteration%25==0): print(iteration,"	",training_loss,"	",validation_loss) 
	
	#RECORD FOR PLOTING
	loss_train.append(training_loss); loss_val.append(validation_loss)
	iterations.append(iteration); iteration+=1

	return training_loss

#INITIAL GUESS
xi=np.random.uniform(0.1,1.,size=NFIT)

#TRAIN MODEL USING SCIPY MINIMIZ 



def optimizer(loss, algo='GD', LR=0.0001, method='batch'):
    global xi, NFIT

    #PARAM
    dx=0.001							#STEP SIZE FOR FINITE DIFFERENCE
    LR=0.0001								#LEARNING RATE
    t=0 	 							#INITIAL ITERATION COUNTER
    tmax=100000							#MAX NUMBER OF ITERATION
    tol=1e-10							#EXIT AFTER CHANGE IN F IS LESS THAN THIS
    

    
    print("INITAL GUESS: ",xi)
    
    while(t<=tmax):
    	t=t+1
    
    	#NUMERICALLY COMPUTE GRADIENT 
    	df_dx=np.zeros(NFIT)
    	for i in range(0,NFIT):
    		dX=np.zeros(NFIT);
    		dX[i]=dx; 
    		xm1=xi-dX; #print(xi,xm1,dX,dX.shape,xi.shape)
    		df_dx[i]=(loss(xi,xt,yt)-loss(xm1,xt,yt))/dx
    	#print(xi.shape,df_dx.shape)
    	xip1=xi-LR*df_dx #STEP 
    
    	if(t%100==0):
    		df=np.mean(np.absolute(loss(xip1,xt,yt)-loss(xi,xt,yt)))
    		print(t,"	",xi,"	","	",loss(xi,xt,yt)) #,df) 
    
    		if(df<tol):
    			print("STOPPING CRITERION MET (STOPPING TRAINING)")
    			break
    
    	#UPDATE FOR NEXT ITERATION OF LOOP
    	xi=xip1
    return xi


def optimizer1(loss, algo='GD', LR=0.0001, method='mini-batch'):
    global xi, NFIT
    
    
    xt1, xt2 ,yt1, yt2= train_test_split(xt,yt, test_size = 0.5)
    
    #PARAM
    dx=0.001							#STEP SIZE FOR FINITE DIFFERENCE
    LR=0.0001								#LEARNING RATE
    t=0 	 							#INITIAL ITERATION COUNTER
    tmax=100000							#MAX NUMBER OF ITERATION
    tol=1e-10							#EXIT AFTER CHANGE IN F IS LESS THAN THIS
    

    
    print("INITAL GUESS: ",xi)
    
    while(t<=tmax):
    	t=t+1
    
    	#NUMERICALLY COMPUTE GRADIENT 
    	df_dx=np.zeros(NFIT)
    	for i in range(0,NFIT):
    		dX=np.zeros(NFIT);
    		dX[i]=dx; 
    		xm1=xi-dX; #print(xi,xm1,dX,dX.shape,xi.shape)
    		df_dx[i]=(loss(xi,xt1,yt1)-loss(xm1,xt1,yt1))/dx
    	#print(xi.shape,df_dx.shape)
    	xip1=xi-LR*df_dx #STEP 
    
    	if(t%100==0):
    		df=np.mean(np.absolute(loss(xip1,xt1,yt1)-loss(xi,xt1,yt1)))
    		print(t,"	",xi,"	","	",loss(xi,xt1,yt1)) #,df) 
    
    		if(df<tol):
    			print("STOPPING CRITERION MET (STOPPING TRAINING)")
    			break
    
    	#UPDATE FOR NEXT ITERATION OF LOOP
    	xi=xip1
    
    t = 0
    while(t<=tmax):
    	t=t+1
    
    	#NUMERICALLY COMPUTE GRADIENT 
    	df_dx=np.zeros(NFIT)
    	for i in range(0,NFIT):
    		dX=np.zeros(NFIT);
    		dX[i]=dx; 
    		xm1=xi-dX; #print(xi,xm1,dX,dX.shape,xi.shape)
    		df_dx[i]=(loss(xi,xt2,yt2)-loss(xm1,xt2,yt2))/dx
    	#print(xi.shape,df_dx.shape)
    	xip1=xi-LR*df_dx #STEP 
    
    	if(t%100==0):
    		df=np.mean(np.absolute(loss(xip1,xt2,yt2)-loss(xi,xt2,yt2)))
    		print(t,"	",xi,"	","	",loss(xi,xt2,yt2)) #,df) 
    
    		if(df<tol):
    			print("STOPPING CRITERION MET (STOPPING TRAINING)")
    			break
    
    	#UPDATE FOR NEXT ITERATION OF LOOP
    	xi=xip1

res = optimizer1(loss, algo='GD', LR=0.0001, method='mini-batch');  popt=res
print("OPTIMAL PARAM:",popt)

#PREDICTIONS
xm=np.array(sorted(xt))
yp=np.array(model(xm,popt))

#UN-NORMALIZE
def unnorm_x(x): 
	return XSTD*x+XMEAN  
def unnorm_y(y): 
	return YSTD*y+YMEAN 

#FUNCTION PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(unnorm_x(xt), unnorm_y(yt), 'o', label='Training set')
	ax.plot(unnorm_x(xv), unnorm_y(yv), 'x', label='Validation set')
	ax.plot(unnorm_x(xm),unnorm_y(yp), '-', label='Model')
	plt.xlabel('x', fontsize=18)
	plt.ylabel('y', fontsize=18)
	plt.legend()
	plt.show()

#PARITY PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(model(xt,popt), yt, 'o', label='Training set')
	ax.plot(model(xv,popt), yv, 'o', label='Validation set')
	plt.xlabel('y predicted', fontsize=18)
	plt.ylabel('y data', fontsize=18)
	plt.legend()
	plt.show()

#MONITOR TRAINING AND VALIDATION LOSS  
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(iterations, loss_train, 'o', label='Training loss')
	ax.plot(iterations, loss_val, 'o', label='Validation loss')
	plt.xlabel('optimizer iterations', fontsize=18)
	plt.ylabel('loss', fontsize=18)
	plt.legend()
	plt.show()

