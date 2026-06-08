# ╔══════════════════════════════════════════════════════════════════════════╗
# ║   TRIKALA DARSHANA · त्रिकाल दर्शन                                      ║
# ║   Step 2+3: SMOTE + Model Training                                      ║
# ║   Trained on Failure. Built for Survival.                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, json, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble          import RandomForestClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import (accuracy_score, f1_score, precision_score,
                                        recall_score, roc_auc_score,
                                        confusion_matrix, classification_report)
from imblearn.over_sampling    import SMOTE
from xgboost                   import XGBClassifier

print("=" * 65)
print("  🪔 TRIKALA DARSHANA — SMOTE + Model Training")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# CELL 1 — LOAD TRAIN-READY DATASET
# ─────────────────────────────────────────────────────────────────
print("\n📂 Loading train-ready dataset...")

df = pd.read_csv('trikala_train_ready.csv')

FEATURES = [
    'TotalFunding','FundingRounds','FundingPerRound',
    'LogFunding','LogFundingPerRound','StageEncoded',
    'CityEncoded','IndustryEncoded','IsMajorCity',
    'IsTechIndustry','FundingBand'
]
TARGET = 'StatusCode'

X = df[FEATURES].fillna(0)
y = df[TARGET]

print(f"✅ Loaded {len(df):,} rows × {len(FEATURES)} features")
print(f"   Safe (1)  : {(y==1).sum():,}")
print(f"   Failed (0): {(y==0).sum():,}")
print(f"   Imbalance ratio: {(y==1).sum()/(y==0).sum():.1f}:1")

# ─────────────────────────────────────────────────────────────────
# CELL 2 — TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CELL 2: Train / Test Split (80/20 Stratified)")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"✅ Train set : {len(X_train):,} rows")
print(f"   Test set  : {len(X_test):,} rows")
print(f"   Train Safe: {(y_train==1).sum():,} | Failed: {(y_train==0).sum():,}")
print(f"   Test  Safe: {(y_test==1).sum():,}  | Failed: {(y_test==0).sum():,}")

# ─────────────────────────────────────────────────────────────────
# CELL 3 — SMOTE
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CELL 3: SMOTE — Balance the Classes")
print("=" * 65)

print("Before SMOTE:")
print(f"  Safe  : {(y_train==1).sum():,}")
print(f"  Failed: {(y_train==0).sum():,}")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(f"  Safe  : {(y_train_sm==1).sum():,}")
print(f"  Failed: {(y_train_sm==0).sum():,}")
print(f"  Total : {len(X_train_sm):,}")
print("✅ Classes balanced!")

# ─────────────────────────────────────────────────────────────────
# CELL 4 — SCALE FEATURES (for Logistic Regression)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CELL 4: Scale Features")
print("=" * 65)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled  = scaler.transform(X_test)
print("✅ StandardScaler applied")

# ─────────────────────────────────────────────────────────────────
# CELL 5 — TRAIN 3 MODELS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CELL 5: Train 3 Models")
print("=" * 65)

results = {}

# ── Model 1: Random Forest ─────────────────────────────────────
print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_sm, y_train_sm)
rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

results['Random Forest'] = {
    'model'    : rf,
    'pred'     : rf_pred,
    'proba'    : rf_proba,
    'accuracy' : accuracy_score(y_test, rf_pred),
    'f1'       : f1_score(y_test, rf_pred),
    'precision': precision_score(y_test, rf_pred),
    'recall'   : recall_score(y_test, rf_pred),
    'roc_auc'  : roc_auc_score(y_test, rf_proba),
}
print(f"  ✅ Accuracy: {results['Random Forest']['accuracy']:.4f}  |  F1: {results['Random Forest']['f1']:.4f}  |  AUC: {results['Random Forest']['roc_auc']:.4f}")

# ── Model 2: XGBoost ──────────────────────────────────────────
print("\n⚡ Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
xgb.fit(X_train_sm, y_train_sm)
xgb_pred  = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]

results['XGBoost'] = {
    'model'    : xgb,
    'pred'     : xgb_pred,
    'proba'    : xgb_proba,
    'accuracy' : accuracy_score(y_test, xgb_pred),
    'f1'       : f1_score(y_test, xgb_pred),
    'precision': precision_score(y_test, xgb_pred),
    'recall'   : recall_score(y_test, xgb_pred),
    'roc_auc'  : roc_auc_score(y_test, xgb_proba),
}
print(f"  ✅ Accuracy: {results['XGBoost']['accuracy']:.4f}  |  F1: {results['XGBoost']['f1']:.4f}  |  AUC: {results['XGBoost']['roc_auc']:.4f}")

# ── Model 3: Logistic Regression ──────────────────────────────
print("\n📈 Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_scaled, y_train_sm)
lr_pred  = lr.predict(X_test_scaled)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

results['Logistic Regression'] = {
    'model'    : lr,
    'pred'     : lr_pred,
    'proba'    : lr_proba,
    'accuracy' : accuracy_score(y_test, lr_pred),
    'f1'       : f1_score(y_test, lr_pred),
    'precision': precision_score(y_test, lr_pred),
    'recall'   : recall_score(y_test, lr_pred),
    'roc_auc'  : roc_auc_score(y_test, lr_proba),
}
print(f"  ✅ Accuracy: {results['Logistic Regression']['accuracy']:.4f}  |  F1: {results['Logistic Regression']['f1']:.4f}  |  AUC: {results['Logistic Regression']['roc_auc']:.4f}")

# ─────────────────────────────────────────────────────────────────
# CELL 6 — COMPARE MODELS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CELL 6: Model Comparison")
print("=" * 65)

print(f"\n{'Model':<22} {'Accuracy':>9} {'F1':>8} {'Precision':>10} {'Recall':>8} {'ROC AUC':>9}")
print("-" * 70)
best_model_name = None
best_f1 = 0
for name, r in results.items():
    marker = " ← BEST" if r['f1'] > best_f1 else ""
    if r['f1'] > best_f1:
        best_f1 = r['f1']
        best_model_name = name
    print(f"{name:<22} {r['accuracy']:>9.4f} {r['f1']:>8.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['roc_auc']:>9.4f}{marker}")

print(f"\n🏆 Best Model: {best_model_name} (F1 = {best_f1:.4f})")

# ─────────────────────────────────────────────────────────────────
# CELL 7 — DETAILED EVALUATION OF BEST MODEL
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"  CELL 7: Detailed Evaluation — {best_model_name}")
print("=" * 65)

best = results[best_model_name]
print("\nClassification Report:")
print(classification_report(y_test, best['pred'], target_names=['Failed','Safe']))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, best['pred'])
print(f"  TN={cm[0][0]:,}  FP={cm[0][1]:,}")
print(f"  FN={cm[1][0]:,}  TP={cm[1][1]:,}")

# Cross validation
print("\n5-Fold Cross Validation:")
best_model_obj = best['model']
cv_X = X_train_scaled if best_model_name == 'Logistic Regression' else X_train_sm
cv_scores = cross_val_score(best_model_obj, cv_X, y_train_sm,
                            cv=StratifiedKFold(5), scoring='f1', n_jobs=-1)
print(f"  F1 scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean F1  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────
# CELL 8 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CELL 8: Visualizations")
print("=" * 65)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor('#07060f')
fig.suptitle('🪔 Trikala Darshana — Model Training Results',
             fontsize=15, color='#FFD700', fontweight='bold', y=0.98)

def style_ax(ax, title):
    ax.set_facecolor('#07060f')
    ax.set_title(title, color='#f0e8d8', fontsize=10, pad=8)
    ax.tick_params(colors='#9a8f7a', labelsize=8)
    ax.xaxis.label.set_color('#9a8f7a')
    ax.yaxis.label.set_color('#9a8f7a')
    for sp in ax.spines.values():
        sp.set_edgecolor('#FFC850')
        sp.set_alpha(0.2)
    ax.grid(color='#FFC850', linestyle='--', linewidth=0.4, alpha=0.08)

model_names  = list(results.keys())
model_colors = ['#FF9933', '#4caf50', '#4fc3f7']

# Plot 1 — Accuracy comparison
ax = axes[0][0]
accs = [results[m]['accuracy'] for m in model_names]
bars = ax.bar(model_names, accs, color=model_colors, edgecolor='none', width=0.5)
for b, v in zip(bars, accs):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
            f'{v:.3f}', ha='center', va='bottom', color='#f0e8d8', fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=8)
style_ax(ax, 'Accuracy Comparison')

# Plot 2 — F1 Score comparison
ax = axes[0][1]
f1s = [results[m]['f1'] for m in model_names]
bars = ax.bar(model_names, f1s, color=model_colors, edgecolor='none', width=0.5)
for b, v in zip(bars, f1s):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
            f'{v:.3f}', ha='center', va='bottom', color='#f0e8d8', fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=8)
style_ax(ax, 'F1 Score Comparison')

# Plot 3 — ROC AUC comparison
ax = axes[0][2]
aucs = [results[m]['roc_auc'] for m in model_names]
bars = ax.bar(model_names, aucs, color=model_colors, edgecolor='none', width=0.5)
for b, v in zip(bars, aucs):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
            f'{v:.3f}', ha='center', va='bottom', color='#f0e8d8', fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=8)
style_ax(ax, 'ROC AUC Comparison')

# Plot 4 — Confusion Matrix (best model)
ax = axes[1][0]
cm = confusion_matrix(y_test, best['pred'])
im = ax.imshow(cm, cmap='RdYlGn', aspect='auto')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Failed','Safe'], color='#f0e8d8', fontsize=9)
ax.set_yticklabels(['Failed','Safe'], color='#f0e8d8', fontsize=9)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                color='#07060f', fontweight='bold', fontsize=11)
style_ax(ax, f'Confusion Matrix — {best_model_name}')

# Plot 5 — Feature Importance (Random Forest)
ax = axes[1][1]
fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
colors_fi = ['#FF9933' if v > fi.mean() else '#4fc3f7' for v in fi.values]
ax.barh(range(len(fi)), fi.values, color=colors_fi, edgecolor='none')
ax.set_yticks(range(len(fi)))
ax.set_yticklabels(fi.index, fontsize=7)
style_ax(ax, 'Feature Importance (Random Forest)')

# Plot 6 — All metrics radar-style bar chart
ax = axes[1][2]
metrics = ['Accuracy','F1','Precision','Recall','ROC AUC']
x = np.arange(len(metrics))
width = 0.25
for i, (name, color) in enumerate(zip(model_names, model_colors)):
    r = results[name]
    vals = [r['accuracy'], r['f1'], r['precision'], r['recall'], r['roc_auc']]
    ax.bar(x + i*width, vals, width, label=name, color=color, alpha=0.85, edgecolor='none')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, rotation=15, ha='right', fontsize=8)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=7, labelcolor='#f0e8d8', facecolor='#07060f', edgecolor='#FFC850')
style_ax(ax, 'All Metrics Comparison')

plt.tight_layout()
plt.savefig('model_results.png', dpi=150, bbox_inches='tight',
            facecolor='#07060f', edgecolor='none')
plt.show()
print("✅ model_results.png saved!")

# ─────────────────────────────────────────────────────────────────
# CELL 9 — SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CELL 9: Save Best Model")
print("=" * 65)

# Always save Random Forest as primary (most interpretable)
# and XGBoost as secondary
joblib.dump(rf,     'trikala_rf_model.pkl')
joblib.dump(xgb,    'trikala_xgb_model.pkl')
joblib.dump(scaler, 'trikala_scaler.pkl')

# Save model accuracy metadata
metadata = {
    'best_model'       : best_model_name,
    'features'         : FEATURES,
    'random_forest'    : {
        'accuracy' : round(results['Random Forest']['accuracy'], 4),
        'f1'       : round(results['Random Forest']['f1'], 4),
        'precision': round(results['Random Forest']['precision'], 4),
        'recall'   : round(results['Random Forest']['recall'], 4),
        'roc_auc'  : round(results['Random Forest']['roc_auc'], 4),
    },
    'xgboost'          : {
        'accuracy' : round(results['XGBoost']['accuracy'], 4),
        'f1'       : round(results['XGBoost']['f1'], 4),
        'precision': round(results['XGBoost']['precision'], 4),
        'recall'   : round(results['XGBoost']['recall'], 4),
        'roc_auc'  : round(results['XGBoost']['roc_auc'], 4),
    },
    'logistic_regression': {
        'accuracy' : round(results['Logistic Regression']['accuracy'], 4),
        'f1'       : round(results['Logistic Regression']['f1'], 4),
        'precision': round(results['Logistic Regression']['precision'], 4),
        'recall'   : round(results['Logistic Regression']['recall'], 4),
        'roc_auc'  : round(results['Logistic Regression']['roc_auc'], 4),
    },
    'cv_mean_f1'       : round(float(cv_scores.mean()), 4),
    'cv_std_f1'        : round(float(cv_scores.std()), 4),
    'train_rows'       : len(X_train_sm),
    'test_rows'        : len(X_test),
}

with open('trikala_model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Saved:")
print("   trikala_rf_model.pkl        — Random Forest model")
print("   trikala_xgb_model.pkl       — XGBoost model")
print("   trikala_scaler.pkl          — StandardScaler")
print("   trikala_model_metadata.json — accuracy metrics")

# ─────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────
rf_acc = results['Random Forest']['accuracy']
xgb_acc = results['XGBoost']['accuracy']

print(f"""
╔══════════════════════════════════════════════════════════════╗
║  MODEL TRAINING SUMMARY                                      ║
╠══════════════════════════════════════════════════════════════╣
║  Training rows (after SMOTE) : {len(X_train_sm):,}                     
║  Test rows                   : {len(X_test):,}                       
║                                                              ║
║  RANDOM FOREST                                               ║
║    Accuracy  : {results['Random Forest']['accuracy']:.4f}                                   
║    F1 Score  : {results['Random Forest']['f1']:.4f}                                   
║    ROC AUC   : {results['Random Forest']['roc_auc']:.4f}                                   
║                                                              ║
║  XGBOOST                                                     ║
║    Accuracy  : {results['XGBoost']['accuracy']:.4f}                                   
║    F1 Score  : {results['XGBoost']['f1']:.4f}                                   
║    ROC AUC   : {results['XGBoost']['roc_auc']:.4f}                                   
║                                                              ║
║  LOGISTIC REGRESSION                                         ║
║    Accuracy  : {results['Logistic Regression']['accuracy']:.4f}                                   
║    F1 Score  : {results['Logistic Regression']['f1']:.4f}                                   
║    ROC AUC   : {results['Logistic Regression']['roc_auc']:.4f}                                   
║                                                              ║
║  🏆 Best Model : {best_model_name:<20}                  
║  Cross-Val F1  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}               
║                                                              ║
║  Next: Run 04_FastAPI_Update.py                              ║
╚══════════════════════════════════════════════════════════════╝
""")
