"""
04_validacion_pi13.py — Validación del modelo (PI13). Solo registros REAL (n=69).
Métrica primaria: F1 macro. Validación cruzada estratificada 5 pliegues x 10 repeticiones (semilla 42).
SMOTE-NC solo dentro del entrenamiento; ordinales redondeadas a niveles válidos; asserts de integridad.
Predictores M1 = socioeconómicos + admisión (sin variables académicas solapadas con la etiqueta).
M2 = todas las variables (solo referencia del solapamiento).
Prueba: t remuestreada corregida (Nadeau y Bengio) con corrección de Holm. n_jobs=1 (reproducible).
"""
import warnings, json, numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
SEED=42; OUT="evidencias/"
import os; os.makedirs(OUT,exist_ok=True)
df=pd.read_csv("dataset_limpio.csv")
real=df[df["origen_dato"]=="REAL"].reset_index(drop=True)
assert len(real)==69 and (real["origen_dato"]=="REAL").all()
y=real["target"].values; X2=real.drop(columns=["target","origen_dato"])
ACAD=["ppa_enc","pct_cursos_aprobados_enc","num_cursos_pendientes_enc","ciclos_atraso_enc","pend_ciclo_anterior","pend_2_ciclos","pend_3_o_mas_ciclos","rendimiento_cursos_criticos"]
X1=X2.drop(columns=ACAD)
print("Clases reales:",np.bincount(y),"| M1:",X1.shape[1],"vars | M2:",X2.shape[1],"vars")
def smote_valid(X,yy,Xcols):
    cat=[i for i,c in enumerate(Xcols) if X[c].nunique()<=2]
    k=max(1,min(5,np.bincount(yy).min()-1))
    sm=SMOTENC(categorical_features=cat,k_neighbors=k,random_state=SEED)
    Xr,yr=sm.fit_resample(X.values.astype(float),yy)
    Xr=pd.DataFrame(Xr,columns=Xcols)
    viol=0
    for i,c in enumerate(Xcols):
        if i in cat: continue
        lv=np.sort(X[c].unique()); idx=np.abs(Xr[c].values[:,None]-lv[None,:]).argmin(1); new=lv[idx]
        viol+=int((np.abs(new-Xr[c].values)>1e-9).sum()); Xr[c]=new
    if "trabaja_enc" in Xcols and "horas_trabajo_enc" in Xcols:
        bad=(Xr["trabaja_enc"]==0)&(Xr["horas_trabajo_enc"]>0); viol+=int(bad.sum()); Xr.loc[bad,"horas_trabajo_enc"]=0
    for c in Xcols:   # assert de valores válidos
        assert set(Xr[c].unique())<=set(X[c].unique())|{0,1} or Xr[c].nunique()<=X[c].nunique()+0, c
    assert len(Xr)>=len(X)
    return Xr,yr,viol
def mk(name):
    if name=="Mayoritaria": return DummyClassifier(strategy="most_frequent")
    if name=="Aleatoria": return DummyClassifier(strategy="stratified",random_state=SEED)
    if name.startswith("LogReg"): return make_pipeline(StandardScaler(),LogisticRegression(class_weight="balanced",max_iter=3000))
    if name.startswith("RF"): return RandomForestClassifier(n_estimators=200,random_state=SEED,n_jobs=1,class_weight="balanced_subsample" if name.endswith("CW") else None)
    if name.startswith("XGB"): return XGBClassifier(n_estimators=200,max_depth=4,learning_rate=0.1,random_state=SEED,n_jobs=1,eval_metric="mlogloss",verbosity=0)
SPECS=[("Mayoritaria",1,False),("Aleatoria",1,False),("LogReg_M1",1,False),("RF_M1_SMOTE",1,True),("RF_M1_CW",1,False),("XGB_M1_SMOTE",1,True),("LogReg_M2",2,False),("RF_M2_SMOTE",2,True)]
def metr(yt,p):
    f=f1_score(yt,p,average=None,labels=[0,1,2],zero_division=0)
    return dict(f1_macro=f.mean(),accuracy=accuracy_score(yt,p),f1_riesgo=f[0],f1_regular=f[1],f1_buen=f[2],
                recall_riesgo=recall_score(yt,p,labels=[0],average="macro",zero_division=0),
                precision_riesgo=precision_score(yt,p,labels=[0],average="macro",zero_division=0))
def run(save_imp=False,reps=10):
    cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=reps,random_state=SEED)
    rows=[];imps=[];viol_tot=0;nsm=0
    for fi,(tr,te) in enumerate(cv.split(X2,y)):
        assert set(real.loc[te,"origen_dato"])=={"REAL"}
        for name,v,sm in SPECS:
            X=(X1 if v==1 else X2); cols=list(X.columns)
            Xtr,ytr=X.iloc[tr],y[tr]
            if sm:
                Xtr,ytr,viol=smote_valid(Xtr,ytr,cols); viol_tot+=viol; nsm+=1
            m=mk(name).fit(Xtr,ytr); p=m.predict(X.iloc[te])
            r=metr(y[te],p); r.update(fold=fi,modelo=name); rows.append(r)
            if save_imp and name=="RF_M1_SMOTE":
                pi=permutation_importance(m,X.iloc[te],y[te],scoring="f1_macro",n_repeats=3,random_state=SEED,n_jobs=1)
                imps.append(pd.Series(pi.importances_mean,index=cols))
    return pd.DataFrame(rows),imps,viol_tot,nsm
res,imps,viol,nsm=run(save_imp=True)
res.to_csv(OUT+"resultados_por_pliegue.csv",index=False)
summ=res.groupby("modelo").agg(["mean","std"]).drop(columns="fold")
summ.columns=[f"{a}_{b}" for a,b in summ.columns]; summ=summ.round(4)
summ.to_csv(OUT+"resultados_pi13_resumen.csv"); print(summ[["f1_macro_mean","f1_macro_std","accuracy_mean","recall_riesgo_mean","precision_riesgo_mean","f1_riesgo_mean","f1_regular_mean","f1_buen_mean"]])
print("Violaciones corregidas tras SMOTE-NC (valores no válidos redondeados):",viol,"en",nsm,"remuestreos")
K=res["fold"].nunique(); ratio=1/4
def nb(a,b):
    d=(res[res.modelo==a].sort_values("fold")["f1_macro"].values-res[res.modelo==b].sort_values("fold")["f1_macro"].values)
    s=np.sqrt((1/K+ratio)*d.var(ddof=1)); t=d.mean()/s if s>0 else np.inf; p=stats.t.sf(t,K-1)
    h=stats.t.ppf(0.975,K-1)*s; dz=d.mean()/d.std(ddof=1) if d.std(ddof=1)>0 else np.inf
    return dict(comparacion=f"{a} - {b}",dif_media=d.mean(),IC95_inf=d.mean()-h,IC95_sup=d.mean()+h,t=t,p_unilateral=p,d_z=dz)
prim=[("RF_M1_SMOTE","Mayoritaria"),("RF_M1_SMOTE","Aleatoria"),("RF_M1_SMOTE","LogReg_M1"),("XGB_M1_SMOTE","Mayoritaria"),("XGB_M1_SMOTE","Aleatoria"),("XGB_M1_SMOTE","LogReg_M1")]
T=pd.DataFrame([nb(a,b) for a,b in prim])
o=np.argsort(T.p_unilateral.values); adj=np.empty(len(T)); run_max=0
for rank,i in enumerate(o):
    run_max=max(run_max,(len(T)-rank)*T.p_unilateral.values[i]); adj[i]=min(1,run_max)
T["p_Holm"]=adj; T["decision(alpha=0.05)"]=np.where(T.p_Holm<0.05,"Se rechaza H0","No se rechaza H0")
T["familia"]="confirmatoria (Holm)"
D=pd.DataFrame([nb("RF_M1_SMOTE","RF_M1_CW"),nb("RF_M2_SMOTE","RF_M1_SMOTE"),nb("LogReg_M2","LogReg_M1")]); D["familia"]="descriptiva (sin decisión)"
pd.concat([T,D]).round(4).to_csv(OUT+"pruebas_estadisticas.csv",index=False); print(pd.concat([T,D]).round(4).to_string())
I=pd.DataFrame(imps); rng=np.random.default_rng(SEED)
bs=np.array([I.values[rng.integers(0,len(I),len(I))].mean(0) for _ in range(2000)])
imp=pd.DataFrame({"variable":I.columns,"importancia_media":I.mean().values,"IC95_inf":np.percentile(bs,2.5,0),"IC95_sup":np.percentile(bs,97.5,0)}).sort_values("importancia_media",ascending=False)
imp["IC_excluye_0"]=(imp.IC95_inf>0)|(imp.IC95_sup<0); imp.round(4).to_csv(OUT+"importancia_M1.csv",index=False); print(imp.head(10).round(4).to_string(index=False))
# CP09 reproducibilidad
a,_,_,_=run(reps=2); b,_,_,_=run(reps=2); diff=float(np.abs(a.drop(columns="modelo").values-b.drop(columns="modelo").values).max()); print("CP09 diferencia absoluta máxima entre dos ejecuciones:",diff)
# CP10 privacidad
x=pd.read_excel("dataset_sipiera_500_real_y_sintetico.xlsx",sheet_name="Respuestas_Formulario")
pat=["nombre","correo","email","dni","código","codigo","telefono","teléfono","celular","apellido"]
hit=[c for c in x.columns if any(p in str(c).lower() for p in pat)]; print("CP10 columnas con posibles identificadores:",hit)
json.dump(dict(viol=viol,nsm=nsm,cp09=diff,cp10=hit,K=K,clases=np.bincount(y).tolist()),open(OUT+"integridad.json","w"))
# figuras
fig,ax=plt.subplots(figsize=(8,4.5)); g=res.groupby("modelo")["f1_macro"]; order=g.mean().sort_values().index
ax.barh(order,g.mean()[order],xerr=1.96*g.std()[order]/np.sqrt(K)*np.sqrt(1+K*ratio),color="#4c72b0"); ax.set_xlabel("F1 macro (media; barra = IC 95 % corregido)"); ax.set_title("F1 macro por modelo (CV 5x10, 69 reales)"); plt.tight_layout(); plt.savefig(OUT+"fig_f1_macro.png",dpi=150); plt.close()
t10=imp.head(10).iloc[::-1]; fig,ax=plt.subplots(figsize=(8,4.5)); ax.barh(t10.variable,t10.importancia_media,xerr=[t10.importancia_media-t10.IC95_inf,t10.IC95_sup-t10.importancia_media],color="#55a868"); ax.set_title("Importancia por permutación en M1 (IC 95 % bootstrap)"); plt.tight_layout(); plt.savefig(OUT+"fig_importancia_M1.png",dpi=150); plt.close()
