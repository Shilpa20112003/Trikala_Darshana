# ╔══════════════════════════════════════════════════════════════════════════╗
# ║   TRIKALA DARSHANA · त्रिकाल दर्शन                                      ║
# ║   Step 1: Feature Engineering Notebook                                  ║
# ║   Trained on Failure. Built for Survival.                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("  🪔 TRIKALA DARSHANA — Feature Engineering")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# CELL 1 — LOAD MASTER DATASET
# ─────────────────────────────────────────────────────────────────
print("\n📂 Loading master dataset...")

df = pd.read_csv('trikala_master_v2.csv')

print(f"✅ Loaded {len(df):,} rows × {df.shape[1]} columns")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nStatus distribution:")
print(df['Status'].value_counts())
print(f"\nSample data:")
print(df.head(5).to_string())

# ─────────────────────────────────────────────────────────────────
# CELL 2 — CLEAN STAGE COLUMN
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CELL 2: Clean & Encode Stage")
print("=" * 65)

def encode_stage(stage):
    """Convert messy stage names to clean numeric encoding"""
    stage = str(stage).lower().strip()
    
    if any(x in stage for x in ['pre-seed', 'pre seed', 'angel', 'friends']):
        return 1  # Pre-seed / Angel
    elif any(x in stage for x in ['seed']):
        return 2  # Seed
    elif any(x in stage for x in ['pre-series a', 'pre series a', 'bridge']):
        return 3  # Bridge / Pre-Series A
    elif 'series a' in stage:
        return 4  # Series A
    elif 'series b' in stage:
        return 5  # Series B
    elif 'series c' in stage:
        return 6  # Series C
    elif any(x in stage for x in ['series d', 'series e', 'series f']):
        return 7  # Late Stage
    elif any(x in stage for x in ['private equity', 'pe']):
        return 8  # Private Equity
    elif any(x in stage for x in ['ipo', 'public']):
        return 9  # IPO
    elif any(x in stage for x in ['debt', 'loan']):
        return 2  # Treat debt as early stage
    elif stage in ['registered']:
        return 2  # MCA registered = treat as seed level
    else:
        return 2  # Unknown = seed level default

df['StageEncoded'] = df['Stage'].apply(encode_stage)

print("\nStage encoding distribution:")
stage_labels = {
    1: 'Pre-seed/Angel', 2: 'Seed/Early', 3: 'Bridge/Pre-A',
    4: 'Series A', 5: 'Series B', 6: 'Series C',
    7: 'Late Stage', 8: 'Private Equity', 9: 'IPO'
}
for code, label in stage_labels.items():
    count = len(df[df['StageEncoded'] == code])
    if count > 0:
        print(f"  {code} — {label}: {count:,}")

# ─────────────────────────────────────────────────────────────────
# CELL 3 — CLEAN CITY + ENCODE
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CELL 3: Clean City & Encode")
print("=" * 65)

# Consolidate city variations
city_fixes = {
    'Bengaluru': 'Bangalore',
    'New Delhi': 'Delhi',
    'Gurugram': 'Gurgaon',
    'Ncr': 'Delhi',
    'Delhi Ncr': 'Delhi',
}
df['City'] = df['City'].replace(city_fixes)

# Keep top 15 cities, group rest as 'Other'
top_cities = df['City'].value_counts().head(15).index.tolist()
df['City'] = df['City'].apply(lambda x: x if x in top_cities else 'Other')

# Encode
le_city = LabelEncoder()
df['CityEncoded'] = le_city.fit_transform(df['City'])

print(f"✅ Cities after cleaning:")
print(df['City'].value_counts().head(10))
print(f"\nCity encoding sample:")
city_map = dict(zip(le_city.classes_, le_city.transform(le_city.classes_)))
for city, code in list(city_map.items())[:8]:
    print(f"  {city}: {code}")

# ─────────────────────────────────────────────────────────────────
# CELL 4 — CLEAN INDUSTRY + ENCODE
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CELL 4: Clean Industry & Encode")
print("=" * 65)

# Map messy industries to clean categories
def clean_industry_label(ind):
    ind = str(ind).lower().strip()
    if any(x in ind for x in ['edtech', 'education', 'learning', 'training']):
        return 'EdTech'
    elif any(x in ind for x in ['fintech', 'finance', 'financial', 'banking', 'payment', 'lending']):
        return 'FinTech'
    elif any(x in ind for x in ['health', 'medical', 'pharma', 'biotech', 'wellness']):
        return 'HealthTech'
    elif any(x in ind for x in ['ecommerce', 'e-commerce', 'retail', 'trading', 'commerce']):
        return 'E-Commerce'
    elif any(x in ind for x in ['saas', 'software', 'technology', 'tech', 'internet', 'consumer internet']):
        return 'Tech/SaaS'
    elif any(x in ind for x in ['food', 'foodtech', 'restaurant', 'delivery', 'foodstuff']):
        return 'FoodTech'
    elif any(x in ind for x in ['logistic', 'transport', 'supply chain', 'storage']):
        return 'Logistics'
    elif any(x in ind for x in ['real estate', 'proptech', 'renting', 'construction']):
        return 'RealEstate'
    elif any(x in ind for x in ['agriculture', 'agri', 'allied activities']):
        return 'AgriTech'
    elif any(x in ind for x in ['media', 'entertainment', 'content', 'news']):
        return 'Media'
    elif any(x in ind for x in ['manufacturing', 'machinery', 'metals', 'chemicals']):
        return 'Manufacturing'
    elif any(x in ind for x in ['business services', 'consulting', 'community', 'personal', 'social']):
        return 'Services'
    elif any(x in ind for x in ['travel', 'tourism', 'hospitality']):
        return 'Travel'
    else:
        return 'Other'

df['IndustryClean'] = df['Industry'].apply(clean_industry_label)

le_industry = LabelEncoder()
df['IndustryEncoded'] = le_industry.fit_transform(df['IndustryClean'])

print("✅ Industry distribution after cleaning:")
print(df['IndustryClean'].value_counts())

# ─────────────────────────────────────────────────────────────────
# CELL 5 — NUMERIC FEATURES
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CELL 5: Create Numeric Features")
print("=" * 65)

# Cap extreme outliers in TotalFunding (top 1%)
funding_cap = df['TotalFunding'].quantile(0.99)
df['TotalFunding'] = df['TotalFunding'].clip(upper=funding_cap)

# FundingPerRound
df['FundingRounds'] = df['FundingRounds'].fillna(1).clip(lower=1)
df['FundingPerRound'] = df['TotalFunding'] / df['FundingRounds']

# Log transform funding (reduces skewness for ML)
df['LogFunding'] = np.log1p(df['TotalFunding'])
df['LogFundingPerRound'] = np.log1p(df['FundingPerRound'])

# IsMajorCity — Tier 1 cities tend to have better survival
major_cities = ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Chennai', 'Pune']
df['IsMajorCity'] = df['City'].apply(lambda x: 1 if x in major_cities else 0)

# IsTechIndustry — Tech startups tend to get more funding
tech_industries = ['EdTech', 'FinTech', 'HealthTech', 'Tech/SaaS', 'E-Commerce']
df['IsTechIndustry'] = df['IndustryClean'].apply(lambda x: 1 if x in tech_industries else 0)

# FundingBand — categorize funding into bands
def funding_band(amount):
    if amount == 0:
        return 0      # No funding
    elif amount < 100000:
        return 1      # < $100K (very early)
    elif amount < 500000:
        return 2      # $100K–500K (seed range)
    elif amount < 2000000:
        return 3      # $500K–2M (Series A territory)
    elif amount < 10000000:
        return 4      # $2M–10M (Series B territory)
    else:
        return 5      # > $10M (growth stage)

df['FundingBand'] = df['TotalFunding'].apply(funding_band)

print("✅ New features created:")
print(f"  FundingPerRound    — avg: ${df['FundingPerRound'].mean():,.0f}")
print(f"  LogFunding         — range: {df['LogFunding'].min():.1f} to {df['LogFunding'].max():.1f}")
print(f"  IsMajorCity        — {df['IsMajorCity'].sum():,} major city startups")
print(f"  IsTechIndustry     — {df['IsTechIndustry'].sum():,} tech startups")
print(f"  FundingBand        — distribution:\n{df['FundingBand'].value_counts().sort_index()}")
print(f"  StageEncoded       — range: {df['StageEncoded'].min()} to {df['StageEncoded'].max()}")

# ─────────────────────────────────────────────────────────────────
# CELL 6 — FINAL FEATURE SET
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CELL 6: Final Feature Set")
print("=" * 65)

# Features to use for ML
FEATURES = [
    'TotalFunding',
    'FundingRounds',
    'FundingPerRound',
    'LogFunding',
    'LogFundingPerRound',
    'StageEncoded',
    'CityEncoded',
    'IndustryEncoded',
    'IsMajorCity',
    'IsTechIndustry',
    'FundingBand',
]

TARGET = 'StatusCode'

X = df[FEATURES]
y = df[TARGET]

print(f"✅ Feature matrix shape: {X.shape}")
print(f"   Target distribution:")
print(f"   Safe   (1): {(y==1).sum():,}")
print(f"   Failed (0): {(y==0).sum():,}")
print(f"\n📋 Final feature list:")
for i, f in enumerate(FEATURES, 1):
    print(f"   {i:02d}. {f}")

# Check for any remaining nulls
print(f"\n🔍 Missing values check:")
nulls = X.isnull().sum()
if nulls.sum() == 0:
    print("   ✅ Zero missing values — clean dataset!")
else:
    print(nulls[nulls > 0])
    X = X.fillna(0)
    print("   ✅ Filled remaining nulls with 0")

# ─────────────────────────────────────────────────────────────────
# CELL 7 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CELL 7: Visualizations")
print("=" * 65)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor('#07060f')
fig.suptitle('🪔 Trikala Darshana — Feature Analysis', 
             fontsize=16, color='#FFD700', fontweight='bold', y=0.98)

colors_safe = '#4caf50'
colors_fail = '#ef5350'

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

# Plot 1 — Status Distribution
ax = axes[0][0]
counts = df['Status'].value_counts()
bars = ax.bar(counts.index, counts.values, color=[colors_safe, colors_fail], edgecolor='none', width=0.5)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 200,
            f'{int(b.get_height()):,}', ha='center', va='bottom', color='#f0e8d8', fontsize=9)
style_ax(ax, 'Status Distribution')

# Plot 2 — Funding Band vs Status
ax = axes[0][1]
band_labels = {0:'No Funding', 1:'<100K', 2:'100K-500K', 3:'500K-2M', 4:'2M-10M', 5:'>10M'}
band_counts = df.groupby(['FundingBand', 'Status']).size().unstack(fill_value=0)
if 'Safe' in band_counts.columns:
    ax.bar(range(len(band_counts)), band_counts.get('Safe', 0), label='Safe', color=colors_safe, alpha=0.85)
if 'Failed' in band_counts.columns:
    ax.bar(range(len(band_counts)), band_counts.get('Failed', 0),
           bottom=band_counts.get('Safe', 0), label='Failed', color=colors_fail, alpha=0.85)
ax.set_xticks(range(len(band_counts)))
ax.set_xticklabels([band_labels.get(i, str(i)) for i in band_counts.index], rotation=30, ha='right', fontsize=7)
style_ax(ax, 'Funding Band vs Status')
ax.legend(fontsize=7, labelcolor='#f0e8d8', facecolor='#07060f', edgecolor='#FFC850')

# Plot 3 — Industry vs Survival Rate
ax = axes[0][2]
ind_survival = df.groupby('IndustryClean')['StatusCode'].mean().sort_values(ascending=False)
colors_bar = ['#4caf50' if v > 0.7 else '#FF9933' if v > 0.5 else '#ef5350' for v in ind_survival.values]
ax.barh(range(len(ind_survival)), ind_survival.values, color=colors_bar, edgecolor='none')
ax.set_yticks(range(len(ind_survival)))
ax.set_yticklabels(ind_survival.index, fontsize=7)
ax.set_xlim(0, 1.1)
style_ax(ax, 'Survival Rate by Industry')

# Plot 4 — City vs Survival Rate
ax = axes[1][0]
city_survival = df.groupby('City')['StatusCode'].mean().sort_values(ascending=False).head(12)
colors_city = ['#4caf50' if v > 0.8 else '#FF9933' if v > 0.6 else '#ef5350' for v in city_survival.values]
ax.barh(range(len(city_survival)), city_survival.values, color=colors_city, edgecolor='none')
ax.set_yticks(range(len(city_survival)))
ax.set_yticklabels(city_survival.index, fontsize=7)
ax.set_xlim(0, 1.1)
style_ax(ax, 'Survival Rate by City')

# Plot 5 — Stage vs Survival Rate
ax = axes[1][1]
stage_survival = df.groupby('StageEncoded')['StatusCode'].mean()
stage_names = [stage_labels.get(i, str(i)) for i in stage_survival.index]
colors_stage = ['#4caf50' if v > 0.8 else '#FF9933' if v > 0.6 else '#ef5350' for v in stage_survival.values]
ax.bar(range(len(stage_survival)), stage_survival.values, color=colors_stage, edgecolor='none', width=0.6)
ax.set_xticks(range(len(stage_survival)))
ax.set_xticklabels(stage_names, rotation=35, ha='right', fontsize=7)
style_ax(ax, 'Survival Rate by Stage')

# Plot 6 — Feature Correlation with Target
ax = axes[1][2]
correlations = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
colors_corr = ['#ef5350' if v < 0 else '#4caf50' for v in correlations.values]
ax.barh(range(len(correlations)), correlations.values, color=colors_corr, edgecolor='none')
ax.set_yticks(range(len(correlations)))
ax.set_yticklabels(correlations.index, fontsize=7)
ax.axvline(x=0, color='#FFC850', linewidth=0.8, alpha=0.5)
style_ax(ax, 'Feature Correlation with Target')

plt.tight_layout()
plt.savefig('feature_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='#07060f', edgecolor='none')
plt.show()
print("✅ Visualization saved as feature_analysis.png")

# ─────────────────────────────────────────────────────────────────
# CELL 8 — SAVE ENGINEERED DATASET
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CELL 8: Save Engineered Dataset")
print("=" * 65)

# Save full engineered dataset
df.to_csv('trikala_engineered.csv', index=False)

# Save just features + target for model training
train_df = df[FEATURES + [TARGET, 'Status']].copy()
train_df.to_csv('trikala_train_ready.csv', index=False)

# Save encoders mapping for use in dashboard/API
import json
city_mapping = dict(zip(le_city.classes_.tolist(), le_city.transform(le_city.classes_).tolist()))
industry_mapping = dict(zip(le_industry.classes_.tolist(), le_industry.transform(le_industry.classes_).tolist()))

mappings = {
    'city_map': city_mapping,
    'industry_map': industry_mapping,
    'features': FEATURES,
    'stage_map': {
        'Pre-seed/Angel': 1, 'Seed': 2, 'Bridge': 3,
        'Series A': 4, 'Series B': 5, 'Series C': 6,
        'Late Stage': 7, 'Private Equity': 8, 'IPO': 9
    }
}
with open('trikala_encodings.json', 'w') as f:
    json.dump(mappings, f, indent=2)

print(f"✅ Saved files:")
print(f"   trikala_engineered.csv   — full dataset with all features ({len(df):,} rows)")
print(f"   trikala_train_ready.csv  — features + target only ({len(train_df):,} rows)")
print(f"   trikala_encodings.json   — city/industry encodings for dashboard")
print(f"\n{'=' * 65}")
print(f"  🪔 Feature Engineering Complete!")
print(f"  Total features: {len(FEATURES)}")
print(f"  Ready for Step 2: SMOTE + Model Training")
print(f"{'=' * 65}")

# ─────────────────────────────────────────────────────────────────
# CELL 9 — SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════════════╗
║  FEATURE ENGINEERING SUMMARY                                 ║
╠══════════════════════════════════════════════════════════════╣
║  Dataset rows        : {len(df):,}                              
║  Total features      : {len(FEATURES)}                                      
║  Safe startups       : {(df['StatusCode']==1).sum():,}                            
║  Failed startups     : {(df['StatusCode']==0).sum():,}                            
║                                                              ║
║  New features added:                                         ║
║  ✅ FundingPerRound   — funding efficiency                   ║
║  ✅ LogFunding        — log-scaled funding                   ║
║  ✅ StageEncoded      — 9-level stage encoding               ║
║  ✅ CityEncoded       — label encoded city                   ║
║  ✅ IndustryEncoded   — 14 clean industry categories         ║
║  ✅ IsMajorCity       — Tier 1 city flag                     ║
║  ✅ IsTechIndustry    — Tech sector flag                     ║
║  ✅ FundingBand       — 6-level funding category             ║
║                                                              ║
║  Next: Run 03_Model_Training_Trikala.py                      ║
╚══════════════════════════════════════════════════════════════╝
""")
