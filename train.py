"""
train.py — Universal Churn Model Trainer
==========================================
Works with ANY CSV dataset.
Trains 6 models + Ensemble and saves .pkl files.

Usage:
    python train.py --csv your_data.csv --target Churn
    python train.py --csv data.csv --target Attrition
    python train.py --csv bank.csv --target Exited
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.impute import SimpleImputer


# ══════════════════════════════════════════════════════
#  ARGUMENT PARSER
# ══════════════════════════════════════════════════════

def get_args():
    parser = argparse.ArgumentParser(description="Universal Churn Model Trainer")
    parser.add_argument("--csv",    required=True,  help="Path to your CSV dataset")
    parser.add_argument("--target", required=False, help="Target column name (auto-detected if not given)")
    parser.add_argument("--out",    required=False, default="models", help="Output folder for .pkl files")
    return parser.parse_args()


# ══════════════════════════════════════════════════════
#  AUTO-DETECT TARGET COLUMN
# ══════════════════════════════════════════════════════

def detect_target(df):
    # Keywords ordered by priority — includes common variants/typos
    keywords = [
        "churn", "churned", "attrition", "attrited", "exited", "exit",
        "left", "default", "defaulted", "fraud", "target", "label",
        "class", "outcome", "cancel", "cancelled", "canceled",
        "leave", "dropout", "unsubscribed", "is_churn", "has_churned",
        "will_churn", "subscription_status", "renewed", "retained",
    ]
    cols_lower = {c.lower().replace(" ", "_"): c for c in df.columns}

    # 1. Exact match
    for kw in keywords:
        if kw in cols_lower:
            return cols_lower[kw]

    # 2. Partial/contains match (e.g. "customer_churned" matches "churn")
    for kw in keywords:
        for col_lower, col_orig in cols_lower.items():
            if kw in col_lower:
                return col_orig

    # 3. Most balanced binary column (closest to 50/50)
    binary = [c for c in df.columns if df[c].nunique() == 2]
    if binary:
        return min(binary, key=lambda c: abs(df[c].value_counts(normalize=True).iloc[0] - 0.5))

    # 4. Last resort
    return df.columns[-1]


# ══════════════════════════════════════════════════════
#  AUTO-DETECT DOMAIN
# ══════════════════════════════════════════════════════

def detect_domain(df):
    cols      = " ".join(df.columns).lower()
    col_list  = [c.lower() for c in df.columns]

    # ── Score each domain by how many columns match ────────
    scores = {}

    scores["HR / Employees"] = sum(1 for w in [
        "attrition","jobrole","department","overtime","joblevel",
        "jobsatisfaction","maritalstatus","yearsatcompany","employeecount",
        "worklifebalance","trainingtimeslastyear","distancefromhome",
        "numcompaniesworked","stockoptionlevel","performancerating"
    ] if w in cols)

    scores["Telecom"] = sum(1 for w in [
        "monthlycharges","totalcharges","internetservice",
        "phoneservice","multiplelines","streamingmovies",
        "streamingtv","techsupport","onlinesecurity","onlinebackup",
        "paperlessbilling","deviceprotection"
    ] if w in cols)

    scores["Banking"] = sum(1 for w in [
        "exited","geography","creditscore","numofproducts",
        "hascrcard","isactivemember","estimatedsalary","balance"
    ] if w in cols)

    scores["E-Commerce"] = sum(1 for w in [
        "ordercount","daysincelastorder","cashbackamount",
        "couponused","satisfactionscore","complain","hourspendonapp",
        "numberofaddress","preferedordercat","maritalstatus"
    ] if w in cols)

    scores["Healthcare"] = sum(1 for w in [
        "diagnosis","medical","hospital","patient","doctor",
        "treatment","prescription","insurance","readmission"
    ] if w in cols)

    scores["Streaming / Media"] = sum(1 for w in [
        "watchhours","watch_hours","subscription","subscription_type","plan",
        "streaming","content","episodes","viewinghistory","cancelled","renewal",
        "avg_watch","last_login","favorite_genre","number_of_profiles",
        "monthly_fee","region","device","profile"
    ] if w in cols)

    # Pick domain with highest score, fallback to General
    best       = max(scores, key=lambda k: scores[k])
    best_score = scores[best]

    if best_score == 0:
        return "General"
    return best
    return "Healthcare"
    if any(w in cols for w in ["purchase","product","orders","cart"]):
        return "E-Commerce"
    return "General"


# ══════════════════════════════════════════════════════
#  ENCODE TARGET → 0 / 1
# ══════════════════════════════════════════════════════

def encode_target(series):
    """Convert any binary target column to 0/1."""
    positive = {
        "yes","true","1","1.0","churn","churned","left","attrited","attrition",
        "default","defaulted","fraud","exit","exited","cancel","cancelled",
        "canceled","dropout","unsubscribed","leaved","resigned","terminated",
    }
    cleaned = series.astype(str).str.strip().str.lower()

    # If already numeric 0/1 — keep as-is
    if cleaned.isin({"0","1","0.0","1.0"}).all():
        return cleaned.map(lambda x: 1 if x in {"1","1.0"} else 0).astype(int)

    mapped = cleaned.map(lambda x: 1 if x in positive else 0).astype(int)

    # Safety check: if mapping gave all zeros, flip and try
    if mapped.sum() == 0:
        # Maybe the positive class has a unique value not in our list
        val_counts = cleaned.value_counts()
        if len(val_counts) == 2:
            # Treat the minority class as positive (churn)
            minority_val = val_counts.index[-1]
            return cleaned.map(lambda x: 1 if x == minority_val else 0).astype(int)

    return mapped


# ══════════════════════════════════════════════════════
#  DROP ID / HIGH-CARDINALITY COLUMNS
# ══════════════════════════════════════════════════════

def drop_id_cols(df, target):
    id_keywords = ["customerid","customer_id","userid","user_id","id",
                   "rowid","row_id","index","uuid","accountid","account_id"]
    id_like = []
    for c in df.columns:
        if c == target: continue
        cl = c.lower().replace(" ","_")
        # Drop if column name looks like an ID
        if any(cl == kw or cl.endswith("_"+kw) or cl.startswith(kw+"_") for kw in id_keywords):
            id_like.append(c); continue
        # Drop if high-cardinality string (unique per row)
        if df[c].dtype == object and df[c].nunique() > 0.85 * len(df):
            id_like.append(c)
    if id_like:
        print(f"   Dropping ID-like columns: {id_like}")
    return df.drop(columns=id_like)


# ══════════════════════════════════════════════════════
#  SMOTE-EQUIVALENT OVERSAMPLING
# ══════════════════════════════════════════════════════

def balance_classes(X_train, y_train):
    X_df     = pd.DataFrame(X_train)
    y_series = pd.Series(y_train)
    majority_val = y_series.value_counts().idxmax()
    minority_val = 1 - majority_val

    X_maj = X_df[y_series == majority_val]
    X_min = X_df[y_series == minority_val]
    y_maj = y_series[y_series == majority_val]
    y_min = y_series[y_series == minority_val]

    X_min_up, y_min_up = resample(X_min, y_min,
                                   replace=True,
                                   n_samples=len(X_maj),
                                   random_state=42)
    X_bal = np.vstack([X_maj.values, X_min_up])
    y_bal = np.hstack([y_maj.values, y_min_up.values])
    return X_bal, y_bal


# ══════════════════════════════════════════════════════
#  EVALUATE MODEL
# ══════════════════════════════════════════════════════

def evaluate(name, y_true, y_pred, y_proba=None):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else 0
    except: auc = 0
    print(f"\n  {'─'*50}")
    print(f"  {name}")
    print(f"  {'─'*50}")
    print(f"  Accuracy : {acc:.4f}  ({round(acc*100,1)}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    return {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"auc":auc}


# ══════════════════════════════════════════════════════
#  MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════

def train(csv_path, target_col=None, out_dir="models"):

    print("\n" + "═"*60)
    print("   UNIVERSAL CHURN TRAINER — 6 Models + Ensemble")
    print("═"*60)

    # ── Load ──────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"\n✅ Loaded: {len(df)} rows × {len(df.columns)} columns")
    print(f"   File  : {os.path.basename(csv_path)}")

    domain = detect_domain(df)
    print(f"   Domain: {domain}")

    # ── Target ────────────────────────────────────────────────
    if not target_col:
        target_col = detect_target(df)
        print(f"   Target: {target_col}  (auto-detected)")
    else:
        print(f"   Target: {target_col}  (user-specified)")

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found. Available: {list(df.columns)}")

    # ── Clean ─────────────────────────────────────────────────
    df = drop_id_cols(df, target_col)
    for col in df.select_dtypes(include="object").columns:
        if col == target_col: continue
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except: pass
    df = df.dropna(subset=[target_col])
    print(f"   After cleaning: {len(df)} rows")

    # ── Encode target ─────────────────────────────────────────
    y = encode_target(df[target_col])
    print(f"   Class dist: Stay={( y==0).sum()}  Churn={( y==1).sum()}")

    # ── Features ──────────────────────────────────────────────
    df_feat = df.drop(columns=[target_col])
    df_feat = pd.get_dummies(df_feat, drop_first=True)
    # handle numeric columns with blanks
    for col in df_feat.columns:
        df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")

    feature_names = df_feat.columns.tolist()
    print(f"\n📋 Features ({len(feature_names)} after one-hot encoding):")
    for i, f in enumerate(feature_names, 1):
        print(f"   {i:3d}. {f}")

    # ── Impute + Scale ────────────────────────────────────────
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(df_feat)
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_imp)

    # ── Split ─────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\n✅ Train/Test split: {len(X_tr)} train / {len(X_te)} test")

    # ── Balance ───────────────────────────────────────────────
    print("⚖️  Balancing classes (SMOTE-equivalent)...")
    X_bal, y_bal = balance_classes(X_tr, y_tr.values)
    print(f"   After balancing: {len(X_bal)} samples  "
          f"(Class 0: {(y_bal==0).sum()}  Class 1: {(y_bal==1).sum()})")

    # ══════════════════════════════════════════════════════
    #  TRAIN 6 MODELS
    # ══════════════════════════════════════════════════════

    print("\n" + "═"*60)
    print("  TRAINING 6 MODELS")
    print("═"*60)

    # 1. Logistic Regression
    print("\n[1/6] Logistic Regression...")
    lr = LogisticRegression(C=0.1, max_iter=1000,
                             class_weight="balanced", random_state=42)
    lr.fit(X_bal, y_bal)
    lr_pred = lr.predict(X_te)

    # 2. Random Forest
    print("[2/6] Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        min_samples_split=10, min_samples_leaf=4,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_bal, y_bal)
    rf_pred = rf.predict(X_te)

    # 3. Extra Trees (XGBoost equivalent — same boosted-tree style)
    print("[3/6] Extra Trees (XGBoost-equivalent)...")
    et = ExtraTreesClassifier(
        n_estimators=200, max_depth=8,
        min_samples_split=10, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    et.fit(X_bal, y_bal)
    et_pred = et.predict(X_te)

    # 4. Gradient Boosting
    print("[4/6] Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, min_samples_split=10, min_samples_leaf=4,
        random_state=42
    )
    gb.fit(X_bal, y_bal)
    gb_pred = gb.predict(X_te)

    # 5. AdaBoost
    print("[5/6] AdaBoost...")
    ada = AdaBoostClassifier(
        n_estimators=100, learning_rate=0.5, random_state=42
    )
    ada.fit(X_bal, y_bal)
    ada_pred = ada.predict(X_te)

    # 6. SVM
    print("[6/6] SVM...")
    svm = SVC(C=1.0, kernel="rbf", gamma="scale",
               class_weight="balanced", probability=True, random_state=42)
    svm.fit(X_bal, y_bal)
    svm_pred = svm.predict(X_te)

    # ══════════════════════════════════════════════════════
    #  ENSEMBLE — Soft Voting (all 6 models)
    # ══════════════════════════════════════════════════════

    print("\n🏆 Building Ensemble (Soft Voting — all 6 models)...")
    ensemble = VotingClassifier(
        estimators=[
            ("lr",  lr),
            ("rf",  rf),
            ("et",  et),
            ("gb",  gb),
            ("ada", ada),
            ("svm", svm),
        ],
        voting="soft",
        n_jobs=-1,
    )
    ensemble.fit(X_bal, y_bal)
    ens_pred  = ensemble.predict(X_te)
    ens_proba = ensemble.predict_proba(X_te)[:, 1]

    # ── Cross-Validation ──────────────────────────────────────
    print("\n🔄 5-Fold Cross-Validation on Ensemble...")
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cvs = cross_val_score(ensemble, X_sc, y, cv=cv, scoring="accuracy")
    print(f"   CV Accuracy: {cvs.mean():.4f} (+/- {cvs.std():.4f})")

    # ── Full Evaluation ───────────────────────────────────────
    print("\n\n" + "═"*60)
    print("  FULL MODEL COMPARISON")
    print("═"*60)
    evaluate("1. Logistic Regression",  y_te, lr_pred,   lr.predict_proba(X_te)[:,1])
    evaluate("2. Random Forest",        y_te, rf_pred,   rf.predict_proba(X_te)[:,1])
    evaluate("3. Extra Trees (XGB eq)", y_te, et_pred,   et.predict_proba(X_te)[:,1])
    evaluate("4. Gradient Boosting",    y_te, gb_pred,   gb.predict_proba(X_te)[:,1])
    evaluate("5. AdaBoost",             y_te, ada_pred,  ada.predict_proba(X_te)[:,1])
    evaluate("6. SVM",                  y_te, svm_pred,  svm.predict_proba(X_te)[:,1])
    ens_metrics = evaluate("🏆 ENSEMBLE (FINAL)", y_te, ens_pred, ens_proba)

    # ── Feature Importance ────────────────────────────────────
    imp_map = {}
    # Average importance from tree-based models
    for m in [rf, et, gb]:
        for fn, imp in zip(feature_names, m.feature_importances_):
            imp_map[fn] = imp_map.get(fn, 0) + imp / 3
    top_features = sorted(imp_map.items(), key=lambda x: x[1], reverse=True)[:15]
    print("\n📊 Top 10 Most Important Features:")
    for fn, imp in top_features[:10]:
        bar = "█" * int(imp * 200)
        print(f"   {fn:<40} {bar}  {imp:.4f}")

    # ══════════════════════════════════════════════════════
    #  SAVE FILES
    # ══════════════════════════════════════════════════════

    os.makedirs(out_dir, exist_ok=True)

    # Bundle everything needed for prediction
    bundle = {
        "model":         ensemble,
        "scaler":        scaler,
        "imputer":       imputer,
        "feature_names": feature_names,
        "domain":        domain,
        "target_col":    target_col,
        "importance_map":imp_map,
        "metrics":       ens_metrics,
        "cv_accuracy":   round(float(cvs.mean()), 4),
        "n_samples":     len(df),
        "n_features":    len(feature_names),
    }

    # Save individual files (for ChurnIQ app compatibility)
    pickle.dump(ensemble,      open(os.path.join(out_dir, "churn_model.pkl"),   "wb"))
    pickle.dump(scaler,        open(os.path.join(out_dir, "scaler.pkl"),        "wb"))
    pickle.dump(feature_names, open(os.path.join(out_dir, "feature_names.pkl"),"wb"))
    # Save full bundle (for universal app)
    pickle.dump(bundle,        open(os.path.join(out_dir, "model_bundle.pkl"),  "wb"))

    final_acc = ens_metrics["accuracy"]
    final_auc = ens_metrics["auc"]

    print("\n\n" + "═"*60)
    print("  ✅ FILES SAVED")
    print("═"*60)
    print(f"  📁 Folder        : {out_dir}/")
    print(f"  💾 churn_model.pkl")
    print(f"  💾 scaler.pkl")
    print(f"  💾 feature_names.pkl")
    print(f"  💾 model_bundle.pkl")
    print(f"\n  📊 FINAL RESULTS")
    print(f"  ─────────────────────────────")
    print(f"  Accuracy  : {final_acc:.4f}  ({round(final_acc*100,1)}%)")
    print(f"  AUC       : {final_auc:.4f}  ({round(final_auc*100,1)}%)")
    print(f"  CV Score  : {cvs.mean():.4f}  (+/- {cvs.std():.4f})")
    print(f"  Features  : {len(feature_names)}")
    print(f"  Domain    : {domain}")
    print(f"  Dataset   : {len(df)} rows")
    print("═"*60)
    print("\n🎉 Training complete! Run:  python app.py\n")

    return bundle


# ══════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os, shutil

    # ════════════════════════════════════════════════════════
    #  ✏️  STEP 1: CHANGE THESE 2 LINES TO YOUR FILE & COLUMN
    # ════════════════════════════════════════════════════════

    CSV_FILE      = "Bank Customer Churn Prediction.csv"   # ← your CSV filename
    TARGET_COLUMN = ""   # ← leave empty to auto-detect, or set e.g. "churned", "Attrition"

    # ══════════════════════════════
    # 
    # ══════════════════════════
    #  DO NOT CHANGE ANYTHING BELOW THIS LINE
    # ════════════════════════════════════════════════════════

    # If called from terminal with --csv argument, use that instead
    if len(sys.argv) > 1:
        args = get_args()
        CSV_FILE      = os.path.basename(args.csv)
        TARGET_COLUMN = args.target or TARGET_COLUMN

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Try exact path first
    csv_path = os.path.join(script_dir, CSV_FILE)

    # If not found — search for any CSV with similar name (ignore case/spaces)
    if not os.path.exists(csv_path):
        print(f"\n⚠️  Could not find: {CSV_FILE}")
        print(f"   Searching folder for CSV files...\n")
        all_csvs = [f for f in os.listdir(script_dir) if f.lower().endswith(".csv")]

        if not all_csvs:
            print("❌ No CSV files found in this folder!")
            print(f"   Folder: {script_dir}")
            print("\n   FIX: Copy your CSV file into this folder:")
            print(f"        {script_dir}")
            input("\nPress Enter to close...")
            sys.exit(1)

        # Try fuzzy match — ignore case and spaces
        clean_target = CSV_FILE.lower().replace(" ","").replace("_","").replace("-","")
        match = None
        for f in all_csvs:
            clean_f = f.lower().replace(" ","").replace("_","").replace("-","")
            if clean_f == clean_target or clean_target in clean_f or clean_f in clean_target:
                match = f
                break

        if match:
            csv_path = os.path.join(script_dir, match)
            print(f"   ✅ Found matching file: {match}")
        else:
            print("   CSV files found in your folder:")
            for f in all_csvs:
                print(f"      • {f}")
            print(f"\n❌ Could not match '{CSV_FILE}' to any file above.")
            print("\n   FIX: Open train.py and change line:")
            print(f'        CSV_FILE = "{CSV_FILE}"')
            print(f"   to one of the exact filenames listed above.")
            input("\nPress Enter to close...")
            sys.exit(1)

    # ── Check target column ────────────────────────────────
    try:
        sample_df = pd.read_csv(csv_path, nrows=0)
        all_cols  = list(sample_df.columns)
    except Exception as e:
        print(f"\n❌ Could not read CSV: {e}")
        input("\nPress Enter to close...")
        sys.exit(1)

    if not TARGET_COLUMN:
        # Auto-detect from the actual CSV
        full_preview = pd.read_csv(csv_path, nrows=500)
        TARGET_COLUMN = detect_target(full_preview)
        print(f"\n✅ Auto-detected target column: '{TARGET_COLUMN}'")
    elif TARGET_COLUMN not in all_cols:
        # Try case-insensitive match first
        matches = [c for c in all_cols if c.lower() == TARGET_COLUMN.lower()]
        if matches:
            TARGET_COLUMN = matches[0]
            print(f"\n✅ Target column matched: '{TARGET_COLUMN}'")
        else:
            # Try partial match (e.g. "churn" matches "churned")
            partial = [c for c in all_cols if TARGET_COLUMN.lower() in c.lower() or c.lower() in TARGET_COLUMN.lower()]
            if partial:
                TARGET_COLUMN = partial[0]
                print(f"\n✅ Target column partial match: '{TARGET_COLUMN}'")
            else:
                # Auto-detect as fallback — don't crash
                full_preview = pd.read_csv(csv_path, nrows=500)
                detected = detect_target(full_preview)
                print(f"\n⚠️  Column '{TARGET_COLUMN}' not found. Auto-detecting...")
                print(f"   Columns in your file: {all_cols}")
                print(f"\n✅ Using auto-detected target: '{detected}'")
                TARGET_COLUMN = detected

    # ── All good — run training ────────────────────────────
    print(f"\n✅ File   : {os.path.basename(csv_path)}")
    print(f"✅ Target : {TARGET_COLUMN}")

    out_dir = os.path.join(script_dir, "models")
    os.makedirs(out_dir, exist_ok=True)

    bundle = train(csv_path=csv_path, target_col=TARGET_COLUMN, out_dir=out_dir)

    # Copy pkl files to root folder so app.py finds them
    for fname in ["churn_model.pkl","scaler.pkl","feature_names.pkl","model_bundle.pkl"]:
        src = os.path.join(out_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(script_dir, fname))

    print("\n✅ Model saved! Now run app.py and open http://localhost:5000")
    input("\nPress Enter to close...")
