import pandas as pd
import numpy as np
from dateutil import parser
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.optimize import brentq,minimize
import pyswarms as ps
from scipy.special import expit
import random
import sys

# Para poder repetir experimentos
random.seed(42)

def muestraErrores(idx_train,idx_test,df,predtrain,predtest):

    X_train = df["demanda"].values[idx_train].reshape((-1,1))
    X_test = df["demanda"].values[idx_test].reshape((-1,1))
    y_train = df["precio"].values[idx_train]
    y_test = df["precio"].values[idx_test]

    # Error cuadrático medio en la predicción del precio
    ecmprecio_train = np.nanmean((predtrain-y_train)**2)
    ecmprecio_test = np.nanmean((predtest-y_test)**2)
    print("Errores cuadráticos de predicción en train",ecmprecio_train,"y test",ecmprecio_test)

    # Errores absolutos
    absprecio_train = np.nanmean(np.abs(predtrain-y_train))
    absprecio_test = np.nanmean(np.abs(predtest-y_test))
    print("Errores absolutos de predicción en train",absprecio_train,"y test",absprecio_test)

    # Errores relativos
    relprecio_train = np.nanmean(np.abs(predtrain-y_train)/y_train)*100
    relprecio_test = np.nanmean(np.abs(predtest-y_test)/y_test)*100
    print("Errores relativos de predicción en train",relprecio_train,"y test",relprecio_test)

def dibujaCurva(X,y,xrange,xrpred):
    plt.rcParams['figure.figsize'] = [50/2.54, 20/2.54]
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Tahoma'] + \
    plt.rcParams['font.sans-serif']
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.size'] = 12
    plt.scatter(X[:,0],y,c="green")
    plt.plot(xrange[:,0],xrpred,c="red",lw=4)  # FIXME
    plt.title("Predicción con una combinación de no lineales (enjambre)")
    plt.xlabel("Demanda (GW)")
    plt.ylabel("Precio (€)")
    plt.show()


rawdata = pd.read_excel("datos.xlsx",sheet_name=["Datos"],header=1)
rawdata = rawdata["Datos"]

mask = (rawdata['Date'] >= '2020-1-1') & (rawdata['Date'] <= '2020-3-31')
rawdata = rawdata[mask]


# Construimos un dataframe reducido
df = pd.DataFrame()
df["tiempo"] = [date.to_pydatetime() for date in rawdata["Date"]]
# Demanda en GW
df["demanda"] = rawdata["Demanda real (MW)"].values/1000
# Precio en euros/MWh
df["precio"] = rawdata["Precio medio horario componente mercado diario (€/MWh)"].values
df.dropna(inplace=True)

# Modelo sencillo para la relación entre la demanda y el precio
idx_train = [i  for i in range(len(df)) if np.random.rand()<0.5]
idx_test = [i for i in range(len(df)) if not i in idx_train ]

X = df["demanda"].values.reshape((-1,1))
y = df["precio"].values
X_train = df["demanda"].values[idx_train].reshape((-1,1))
X_test = df["demanda"].values[idx_test].reshape((-1,1))
y_train = df["precio"].values[idx_train]
y_test = df["precio"].values[idx_test]
minD = min(X)
maxD = max(X)
xrange = np.linspace(minD,maxD,100).reshape((-1,1))

##########################################################################
# Clustering

Nc = range(1, 7)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Número de Clusters')
plt.ylabel('Puntuación')
plt.title('Curva del codo')
plt.show()

kmeans = KMeans(n_clusters=3).fit(X)
centroids = kmeans.cluster_centers_
print(centroids)

# Prediciendo los clusters
labels = kmeans.predict(X)
print(labels)
# Obteniendo los centros
C = kmeans.cluster_centers_
print(C)
colores=['red','green','blue']
asignar=[]
for row in labels:
    asignar.append(colores[row])
 
fig = plt.figure()
plt.scatter(X,y,c=asignar, s=70)
plt.title("Clustering con 3 clústers")
plt.xlabel("Demanda (GW)")
plt.ylabel("Precio (€)")
plt.show()

#########################################################################


reg = GradientBoostingRegressor(random_state=0).fit(X_train, y_train)
predtrain = reg.predict(X_train)
predtest = reg.predict(X_test)
xrpred = reg.predict(xrange.reshape((-1,1)))
muestraErrores(idx_train,idx_test,df,predtrain,predtest)
print("(DEBUG) Error en train/test/total:",
    np.nansum(abs(y_train-reg.predict(X_train.reshape((-1,1))))),
    np.nansum(abs(y_test-reg.predict(X_test.reshape((-1,1))))),
    np.nansum(abs(y-reg.predict(X.reshape((-1,1))))))
print("Mostrando gráfica de XGBoost")
dibujaCurva(X,y,xrange,xrpred)

# Regresión isotónica
# Forzando a que la curva de precio-cantidad sea creciente
iso_reg = IsotonicRegression().fit(X_train[:,0], y_train)
predtrain = iso_reg.predict(X_train[:,0])
predtest = iso_reg.predict(X_test[:,0])
xrpred = iso_reg.predict(xrange[:,0])
muestraErrores(idx_train,idx_test,df,predtrain,predtest)
print("(DEBUG) Error en todo el conjunto:",
    np.nansum(abs(y_train-iso_reg.predict(X_train[:,0]))),
    np.nansum(abs(y_test-iso_reg.predict(X_test[:,0]))),
    np.nansum(abs(y-iso_reg.predict(X[:,0]))))
print("Mostrando gráfica isotónica")
dibujaCurva(X,y,xrange,xrpred)

# ------------------------------------------------------------------------
# Curvas lineales de oferta y demanda (una única oferta)
# ------------------------------------------------------------------------

def oferta(precio,params):
    # Cantidad ofertada a cada precio
    return params[0]+params[1]*precio

def demanda(precio,params):
    # Cantidad demandada a cada precio
    return params[0]-params[1]*precio

def precio_I(cantidad_real,x):
    myf = lambda precio_modelo: oferta(precio_modelo,x[0:2])-demanda(precio_modelo,[cantidad_real,x[2]])
    lowp = 0
    highp = 100
    if myf(lowp)*myf(highp)<0:
        res = brentq(myf,lowp,highp)
    else:
        # Sin solución en el rango de valores
        res = highp*10
    return res,oferta(res,x[0:2])

def pred_casacion_I(x,df):
    precio_modelo = np.zeros(len(df))
    cantidad_modelo = np.zeros(len(df))
    for index,row in df.iterrows():
        cantidad_real = row["demanda"]
        precio_modelo[index], cantidad_modelo[index] = precio_I(cantidad_real,x)
    return precio_modelo,cantidad_modelo

def casacion_I(x,df):
    precio_modelo,cantidad_modelo = pred_casacion_I(x,df)
    error = 0
    for index,row in df.iterrows():
        cantidad_real = row["demanda"]
        precio_real = row["precio"]
        error_cantidad = cantidad_real - cantidad_modelo[index]
        error_precio = precio_real - precio_modelo[index]
        error += np.sqrt(error_cantidad**2+error_precio**2)
    return error

x = np.array([1.2536077,  0.78503647, 0.        ])
OPTIMIZAR = True
if OPTIMIZAR:
    res = minimize(lambda x:casacion_I(x,df.loc[idx_train,:].reset_index()),x,method="L-BFGS-B",bounds=[(0,np.inf)]*3)
    x = res.x


# Predicciones de cantidad y precio
predtrain, _ = pred_casacion_I(x,df.loc[idx_train,:].reset_index())
predtest, _ = pred_casacion_I(x,df.loc[idx_test,:].reset_index())
xrpred, _ = pred_casacion_I(x,pd.DataFrame({"demanda":xrange[:,0]}))
muestraErrores(idx_train,idx_test,df,predtrain,predtest)
print("(DEBUG) Error en train/test/todo:",casacion_I(x,df.loc[idx_train,:].reset_index()),casacion_I(x,df.loc[idx_test,:].reset_index()),casacion_I(x,df))
print("Mostrando una lineal")
dibujaCurva(X,y,xrange,xrpred)

# --------------------------------------------------------------------------------------------
# Curvas lineales de oferta y demanda (una curva de oferta por agente, con capacidad limitada)
# --------------------------------------------------------------------------------------------

def oferta_II(precio,cuota,params):
    of1 = min(params[0]+params[1]*precio,cuota[0])
    of2 = min(params[2]+params[3]*precio,cuota[1])
    of3 = min(params[4]+params[5]*precio,cuota[2])
    of4 = min(params[6]+params[7]*precio,cuota[3])
    of5 = min(params[8]+params[9]*precio,cuota[4])
    # Cantidad negociada
    return of1+of2+of3+of4+of5

def precio_II(cantidad_real,cuota,x):
    myf = lambda precio_modelo: oferta_II(precio_modelo,cuota,x[0:10])-demanda(precio_modelo,[cantidad_real,x[10]])
    lowp = 0
    highp = 100
    if myf(lowp)*myf(highp)<0:
        res = brentq(myf,lowp,highp)
    else:
        res = 65
    return res,oferta_II(res,cuota,x[0:10])


def pred_casacion_II(x,cuota,df):
    precio_modelo = np.zeros(len(df))
    cantidad_modelo = np.zeros(len(df))
    for index,row in df.iterrows():
        cantidad_real = row["demanda"]
        precio_modelo[index], cantidad_modelo[index] = precio_II(cantidad_real,cuota,x)
    return precio_modelo,cantidad_modelo


def casacion_II(x,cuota,df):
    precio_modelo,cantidad_modelo = pred_casacion_II(x,cuota,df)
    error = 0
    for index,row in df.iterrows():
        cantidad_real = row["demanda"]
        precio_real = row["precio"]
        error_cantidad = cantidad_real - cantidad_modelo[index]
        error_precio = precio_real - precio_modelo[index]
        error += np.sqrt(error_cantidad**2+error_precio**2)
    return error

maxdemanda = max(df["demanda"])
fcuota = np.array([0.20,0.16,0.18,0.06,0.40])
cuota = maxdemanda * fcuota
'''
OPTIMIZAR = "Greedy"
if OPTIMIZAR == "Greedy":
    x = np.array([0, 0.5]*5+[0])
    xtr = df.loc[idx_train,:].reset_index()
    res = minimize(lambda x:casacion_II(x,cuota,xtr),x,method="L-BFGS-B",bounds=[(0,np.inf)]*11)
    x = res.x
elif OPTIMIZAR == "PSO":
    # Mismo problema con PSO
    options = {'c1': 1.50, 'c2': 1.50, 'w':0.73}
    xtr = df.loc[idx_train,:].reset_index()
    def fv(vx):
        return [casacion_II(x,cuota,xtr) for x in vx] 
    bounds=[(0,3)]*11
    optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=11, options=options, bounds=([b[0] for b in bounds],[b[1] for b in bounds]))
    cost, pos = optimizer.optimize(fv, iters=5000)
    x = pos
    res = minimize(lambda x:casacion_II(x,cuota,xtr),x,method="Nelder-Mead",bounds=[(0,np.inf)]*11)
    x = res.x

# Predicciones de cantidad y precio
predtrain, _ = pred_casacion_II(x,cuota,df.loc[idx_train,:].reset_index())
predtest, _ = pred_casacion_II(x,cuota,df.loc[idx_test,:].reset_index())
xrpred, _ = pred_casacion_II(x,cuota,pd.DataFrame({"demanda":xrange[:,0]}))
muestraErrores(idx_train,idx_test,df,predtrain,predtest)
print("Mostrando una combinación de lineales")
dibujaCurva(X,y,xrange,xrpred)
'''
# --------------------------------------------------------------------------------------------
# Curvas no lineales de oferta y demanda (una curva de oferta por agente, con capacidad limitada)
# --------------------------------------------------------------------------------------------

def heaviside(x):
    return expit(x)

def escalon(precio,tecnologias,preciostecno):
    capacidad = 0
    sumpreciostecno = 0
    for i,tec in enumerate(tecnologias):
        sumpreciostecno += preciostecno[i]
        capacidad += tec*heaviside(precio-sumpreciostecno)
    return capacidad

def oferta_III(precio,cuota,ftecnologias,preciostecno):
    of1 = escalon(precio,ftecnologias[0:5]*cuota[0],preciostecno[0:5])
    of2 = escalon(precio,ftecnologias[5:10]*cuota[1],preciostecno[5:10])
    of3 = escalon(precio,ftecnologias[10:15]*cuota[2],preciostecno[10:15])
    of4 = escalon(precio,ftecnologias[15:20]*cuota[3],preciostecno[15:20])
    of5 = escalon(precio,ftecnologias[20:25]*cuota[4],preciostecno[20:25])
    return of1+of2+of3+of4+of5

def precio_III(cantidad_real,cuota,x):
    ftecno = x[0:25]
    ftecno[0:5] /= np.sum(x[0:5])
    ftecno[5:10] /= np.sum(x[5:10])
    ftecno[10:15] /= np.sum(x[10:15])
    ftecno[15:20] /= np.sum(x[15:20])
    ftecno[20:25] /= np.sum(x[20:25])
    preciostecno = x[25:50]*100
    myf = lambda precio_modelo: oferta_III(precio_modelo,cuota,ftecno,preciostecno)-demanda(precio_modelo,[cantidad_real,x[50]])
    lowp = 0
    highp = 100
    if myf(lowp)*myf(highp)<0:
        res = brentq(myf,lowp,highp)
    else:
        res = highp*10
    return res,oferta_III(res,cuota,ftecno,preciostecno)

def pred_casacion_III(x,cuota,df):
    precio_modelo = np.zeros(len(df))
    cantidad_modelo = np.zeros(len(df))
    for index,row in df.iterrows():
        cantidad_real = row["demanda"]
        precio_modelo[index], cantidad_modelo[index] = precio_III(cantidad_real,cuota,x)
    return precio_modelo,cantidad_modelo

def casacion_III(x,cuota,df):
    precio_modelo,cantidad_modelo = pred_casacion_III(x,cuota,df)
    error = 0
    for index,row in df.iterrows():
        cantidad_real = row["demanda"]
        precio_real = row["precio"]
        error_cantidad = cantidad_real - cantidad_modelo[index]
        error_precio = precio_real - precio_modelo[index]
        error += np.sqrt(error_cantidad**2+error_precio**2)
    return error


maxdemanda = max(df["demanda"])
fcuota = np.array([0.20,0.16,0.18,0.06,0.40])
cuota = maxdemanda * fcuota

OPTIMIZAR = "Greedy"
if OPTIMIZAR == "Greedy":
    x = np.array([0.50,0.30,0.18,0.01,0.01]*5+[0.10]*25+[0])
    xtr = df.loc[idx_train,:].reset_index()
    res = minimize(lambda x:casacion_III(x,cuota,xtr),x,method="L-BFGS-B",bounds=[(0,1)]*25+[(0,1),(0,1),(0,1),(0,1),(0,0.2)]*5+[(0,1)])
    x = res.x
elif OPTIMIZAR == "PSO":
    # Mismo problema con PSO
    options = {'c1': 1.50, 'c2': 1.50, 'w':0.73}
    xtr = df.loc[idx_train,:].reset_index()
    def fv(vx):
        return [casacion_III(x,cuota,xtr) for x in vx] 
    bounds=[(0,1)]*51
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=51, options=options, bounds=([b[0] for b in bounds],[b[1] for b in bounds]))
    cost, pos = optimizer.optimize(fv, iters=250)
    x = pos
    res = minimize(lambda x:casacion_III(x,cuota,xtr),x,method="Nelder-Mead",bounds=[(0,1)]*51)
    x = res.x



# Predicciones de cantidad y precio
predtrain, _ = pred_casacion_III(x,cuota,df.loc[idx_train,:].reset_index())
predtest, _ = pred_casacion_III(x,cuota,df.loc[idx_test,:].reset_index())
xrpred, _ = pred_casacion_III(x,cuota,pd.DataFrame({"demanda":xrange[:,0]}))
muestraErrores(idx_train,idx_test,df,predtrain,predtest)
print("Mostrando una combinación de no lineales")
dibujaCurva(X,y,xrange,xrpred)

# Curvas de oferta de los agentes
for agente in range(5):
    pt = x[25+0*agente:25+5*agente]
    tec = x[0*agente:5*agente]*cuota[agente]
    dbgprecio = np.linspace(0,140,100)
    plt.plot(dbgprecio,[escalon(p,tec,pt) for p in dbgprecio])
plt.show()
