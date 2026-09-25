"""Titanic modeling stage. Reads the committed CSV; does not reload seaborn."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay, mean_absolute_error,
                             mean_squared_error, r2_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

ROOT=Path(__file__).resolve().parent; FIG=ROOT/"figures"; OUT=ROOT/"output"; MODELS=ROOT/"models"
for p in [FIG,OUT,MODELS]: p.mkdir(exist_ok=True)
RANDOM_STATE=42


def make_preprocessor(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ])


def metrics(model, X, y):
    pred=model.predict(X); prob=model.predict_proba(X)[:,1]
    return {"accuracy":accuracy_score(y,pred),"precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),"auc":roc_auc_score(y,prob)}


def main():
    df=pd.read_csv(ROOT/"titanic.csv")
    # Model variables intentionally exclude redundant derived flags and target.
    features=["pclass","sex","age","sibsp","parch","fare","embarked"]
    X=df[features]; y=df["survived"]
    class_counts = y.value_counts().sort_index()
    class_percent = (y.value_counts(normalize=True).sort_index() * 100).round(2)
    class_balance_text = (
        f"Not survived (0): {int(class_counts.get(0, 0))} "
        f"({class_percent.get(0, 0.0):.2f}%)\n"
        f"Survived (1): {int(class_counts.get(1, 0))} "
        f"({class_percent.get(1, 0.0):.2f}%)"
    )
    stratification_text = (
        "Stratification preserves approximately the same survived/not-survived "
        "class proportions in both the training and test splits. This matters "
        "because the target classes are imbalanced, so an unstratified split "
        "could produce a less representative test set."
    )

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=RANDOM_STATE)
    num=["pclass","age","sibsp","parch","fare"]; cat=["sex","embarked"]

    models={
        "Logistic Regression":LogisticRegression(max_iter=2000,random_state=RANDOM_STATE),
        "Decision Tree":DecisionTreeClassifier(max_depth=5,random_state=RANDOM_STATE),
        "Random Forest":RandomForestClassifier(n_estimators=300,random_state=RANDOM_STATE),
    }
    results=[]; curves=[]
    for name,est in models.items():
        pipe=Pipeline([("preprocess",make_preprocessor(num,cat)),("model",est)])
        pipe.fit(X_train,y_train); m=metrics(pipe,X_test,y_test); m["model"]=name; results.append(m)
        pred=pipe.predict(X_test); prob=pipe.predict_proba(X_test)[:,1]
        ConfusionMatrixDisplay.from_predictions(y_test,pred); plt.title(f"Confusion matrix — {name}"); plt.tight_layout(); plt.savefig(FIG/f"cm_{name.lower().replace(' ','_')}.png",dpi=160); plt.close()
        fpr,tpr,_=roc_curve(y_test,prob); curves.append((name,fpr,tpr,m["auc"]))
        if name=="Decision Tree":
            feature_names=list(pipe.named_steps["preprocess"].get_feature_names_out())
            plt.figure(figsize=(18,10)); plot_tree(pipe.named_steps["model"],feature_names=feature_names,class_names=["0","1"],filled=False,max_depth=4); plt.tight_layout(); plt.savefig(FIG/"decision_tree.png",dpi=140); plt.close()
    plt.figure(figsize=(7,5))
    for name,fpr,tpr,auc in curves: plt.plot(fpr,tpr,label=f"{name} AUC={auc:.3f}")
    plt.plot([0,1],[0,1],linestyle="--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC comparison"); plt.legend(); plt.tight_layout(); plt.savefig(FIG/"roc_comparison.png",dpi=160); plt.close()
    clf_table=pd.DataFrame(results).set_index("model").round(4); clf_table.to_csv(OUT/"classifier_metrics.csv")

    # Imbalance comparison on identical split. SMOTE occurs only inside training data.
    imbalance=[]
    base=Pipeline([("preprocess",make_preprocessor(num,cat)),("model",LogisticRegression(max_iter=2000,random_state=RANDOM_STATE))]); base.fit(X_train,y_train); b=metrics(base,X_test,y_test); b["strategy"]="baseline"; imbalance.append(b)
    balanced=Pipeline([("preprocess",make_preprocessor(num,cat)),("model",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=RANDOM_STATE))]); balanced.fit(X_train,y_train); b=metrics(balanced,X_test,y_test); b["strategy"]="class_weight=balanced"; imbalance.append(b)
    smote=ImbPipeline([("preprocess",make_preprocessor(num,cat)),("smote",SMOTE(random_state=RANDOM_STATE)),("model",LogisticRegression(max_iter=2000,random_state=RANDOM_STATE))]); smote.fit(X_train,y_train); b=metrics(smote,X_test,y_test); b["strategy"]="SMOTE training fold"; imbalance.append(b)
    imbalance_table = pd.DataFrame(imbalance).set_index("strategy").round(4)
    imbalance_table.to_csv(OUT/"imbalance_comparison.csv")

    imbalance_best = imbalance_table["f1"].idxmax()
    imbalance_best_row = imbalance_table.loc[imbalance_best]
    imbalance_conclusion = (
        f"Imbalance conclusion: {imbalance_best} produced the highest F1 "
        f"({imbalance_best_row.f1:.4f}) on the held-out test split. "
        f"Its recall was {imbalance_best_row.recall:.4f} and precision was "
        f"{imbalance_best_row.precision:.4f}, so on this split it provided the "
        "strongest balance between identifying survivors and limiting false "
        "positive predictions among the three tested strategies."
    )

    # RF tuning with OOB enabled.
    rf=RandomForestClassifier(oob_score=True,random_state=RANDOM_STATE)
    rf_pipe=Pipeline([("preprocess",make_preprocessor(num,cat)),("model",rf)])
    grid=GridSearchCV(rf_pipe,{"model__n_estimators":[100,200],"model__max_depth":[None,5,10],"model__max_features":["sqrt","log2"]},cv=5,scoring="f1",n_jobs=-1)
    grid.fit(X_train,y_train); best=grid.best_estimator_; oob=best.named_steps["model"].oob_score_
    (OUT/"rf_tuning.txt").write_text(f"Best params: {grid.best_params_}\nBest CV F1: {grid.best_score_:.4f}\nOOB score: {oob:.4f}\n",encoding="utf-8")

    # Regression: fare from all other available useful features, with train-only preprocessing.
    reg_features=[c for c in ["pclass","sex","age","sibsp","parch","embarked","alone"] if c in df.columns]
    Xr=df[reg_features]; yr=df["fare"]
    Xr_train,Xr_test,yr_train,yr_test=train_test_split(Xr,yr,test_size=.2,random_state=RANDOM_STATE)
    rnum=[c for c in reg_features if pd.api.types.is_numeric_dtype(df[c])]; rcat=[c for c in reg_features if c not in rnum]
    reg=Pipeline([("preprocess",make_preprocessor(rnum,rcat)),("model",LinearRegression())]); reg.fit(Xr_train,yr_train); pred=reg.predict(Xr_test)
    mae=mean_absolute_error(yr_test,pred); rmse=np.sqrt(mean_squared_error(yr_test,pred)); r2=r2_score(yr_test,pred); n=len(yr_test); p=len(reg.named_steps["preprocess"].get_feature_names_out()); adj=1-(1-r2)*(n-1)/(n-p-1)
    residuals=yr_test-pred
    plt.figure(figsize=(7,4)); plt.scatter(pred,residuals,alpha=.65); plt.axhline(0,linestyle="--"); plt.xlabel("Predicted fare"); plt.ylabel("Residual"); plt.title("Fare regression residual plot"); plt.tight_layout(); plt.savefig(FIG/"fare_residuals.png",dpi=160); plt.close()
    residual_spread_corr = np.corrcoef(np.abs(residuals), pred)[0,1]
    if np.isfinite(residual_spread_corr):
        hetero = (
            "possible heteroscedasticity"
            if abs(residual_spread_corr) > 0.2
            else "no strong evidence of heteroscedasticity"
        )
    else:
        hetero = "could not be assessed from the residual-spread correlation"
    reg_metrics={"MAE":mae,"RMSE":rmse,"R2":r2,"Adjusted_R2":adj,"heteroscedasticity":hetero}
    (OUT/"regression_metrics.txt").write_text("\n".join(f"{k}: {v}" for k,v in reg_metrics.items()),encoding="utf-8")

    # Final complete pipeline artifact: tuned classifier with preprocessing + estimator.
    joblib.dump(best,MODELS/"best_titanic_pipeline.joblib")
    reloaded=joblib.load(MODELS/"best_titanic_pipeline.joblib")
    reload_pred=reloaded.predict(X_test.head(3)); (OUT/"reload_check.txt").write_text(f"Raw-input reload predictions: {reload_pred.tolist()}\n",encoding="utf-8")

    best_row=clf_table.sort_values("f1",ascending=False).iloc[0]
    best_name=clf_table.sort_values("f1",ascending=False).index[0]
    recommendation=(f"Based on the held-out test split, {best_name} has the highest F1 score ({best_row.f1:.4f}) among the three classifiers. "
                    f"Its accuracy is {best_row.accuracy:.4f}, precision is {best_row.precision:.4f}, recall is {best_row.recall:.4f}, and AUC is {best_row.auc:.4f}. "
                    "These metrics should be considered together because accuracy, precision, recall, F1, and AUC capture different aspects of classification performance. "
                    "The selected pipeline is saved with preprocessing and the estimator together so it can accept raw feature rows end-to-end.")
    final=(
        "Class balance and stratification:\n"
        + class_balance_text
        + "\n\n"
        + stratification_text
        + "\n\n"
        + "Classifier comparison:\n"
        + clf_table.to_string()
        + "\n\n"
        + "Imbalance strategy comparison:\n"
        + imbalance_table.to_string()
        + "\n"
        + imbalance_conclusion
        + "\n\n"
        + "Regression metrics:\n"
        + pd.Series(reg_metrics).to_string()
        + "\n\n"
        + "Deployment recommendation:\n"
        + recommendation
        + "\n"
    )
    final = "\n".join(line.rstrip() for line in final.splitlines()) + "\n"
    (OUT/"model_comparison.txt").write_text(final,encoding="utf-8")
    print(clf_table)
    print("Modeling complete. See analytics/output and analytics/models.")

if __name__=="__main__": main()
