import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

# ---------------------------------------------------------
# STEP 1: DATA WRANGLING - Calculate player ages
# ---------------------------------------------------------
df = pd.read_excel('/mnt/user-data/uploads/FIFA_World_Cup_2026_Filtered_Profiles.xlsx')
df['Date of Birth'] = pd.to_datetime(df['Date of Birth'])

# Reference date = opening day of the FIFA World Cup 2026
REF_DATE = pd.Timestamp('2026-06-11')

def calc_age(dob, ref=REF_DATE):
    years = ref.year - dob.year
    if (ref.month, ref.day) < (dob.month, dob.day):
        years -= 1
    return years

df['Age'] = df['Date of Birth'].apply(calc_age)

print("POPULATION SUMMARY (N =", len(df), ")")
print(df[['Player Name','Team','Position','Date of Birth','Age']].head())
print()
print(df.groupby('Position')['Age'].describe())
print()

# ---------------------------------------------------------
# STEP 2: DATA PREPARATION & SAMPLING
# ---------------------------------------------------------
N = len(df)
pop_std_pilot = df['Age'].std(ddof=0)  # population std used purely to justify sample size
Z = 1.96
E = 1.0  # desired margin of error, in years

n0 = (Z * pop_std_pilot / E) ** 2
n_adj = n0 / (1 + (n0 - 1) / N)
n_sample = int(np.ceil(n_adj))
print(f"Population std (pilot) = {pop_std_pilot:.3f}")
print(f"Unadjusted n0 = {n0:.2f}, finite-pop-corrected n = {n_adj:.2f} -> use n = {n_sample}")

np.random.seed(42)
sample = df.sample(n=n_sample, random_state=42).reset_index(drop=True)
print()
print("SAMPLE (n =", len(sample), ")")
print(sample['Position'].value_counts())


# ---------------------------------------------------------
# STEP 3: DESCRIPTIVE STATISTICS (on the sample)
# ---------------------------------------------------------
ages = sample['Age']
desc = {
    'n': len(ages),
    'mean': ages.mean(),
    'median': ages.median(),
    'std': ages.std(ddof=1),
    'variance': ages.var(ddof=1),
    'min': ages.min(),
    'max': ages.max(),
    'range': ages.max() - ages.min(),
    'skew': ages.skew(),
}
print("\n--- DESCRIPTIVE STATISTICS (sample, n=40) ---")
for k, v in desc.items():
    print(f"{k:10s}: {v:.3f}" if isinstance(v, float) else f"{k:10s}: {v}")

# Age distribution (bins)
bins = [16,20,24,28,32,36,40]
sample['AgeGroup'] = pd.cut(sample['Age'], bins=bins)
print("\nAge distribution (sample):")
print(sample['AgeGroup'].value_counts().sort_index())

# ---------------------------------------------------------
# STEP 4: 95% CONFIDENCE INTERVAL FOR POPULATION MEAN AGE
# ---------------------------------------------------------
n = len(ages)
mean = ages.mean()
sem = ages.std(ddof=1) / np.sqrt(n)
t_crit = stats.t.ppf(0.975, df=n-1)
ci_low, ci_high = mean - t_crit*sem, mean + t_crit*sem
print(f"\n95% CI for population mean age: ({ci_low:.3f}, {ci_high:.3f})")
print(f"Sample mean = {mean:.3f}, SEM = {sem:.3f}, t_crit(df={n-1}) = {t_crit:.3f}")

# Sanity check vs actual population mean (we happen to have full population)
print(f"[Check] True population mean age (N=108) = {df['Age'].mean():.3f}  -> ", 
      "inside CI" if ci_low <= df['Age'].mean() <= ci_high else "OUTSIDE CI")

# ---------------------------------------------------------
# STEP 5: ONE-SAMPLE T-TEST -- Midfielders vs Defenders
# ---------------------------------------------------------
# Reference value (mu0) = population mean age of ALL defenders (exhaustively observed, N=38)
def_pop = df[df['Position']=='DEF']['Age']
mu0 = def_pop.mean()
print(f"\nDefender population: N={len(def_pop)}, mean age (mu0) = {mu0:.3f}, std = {def_pop.std(ddof=1):.3f}")

# Draw an independent random sample of MIDFIELDERS to test against mu0
mid_pop = df[df['Position']=='MID']
N_mid = len(mid_pop)
mid_std_pilot = mid_pop['Age'].std(ddof=0)
n0_mid = (Z * mid_std_pilot / E) ** 2
n_mid_adj = n0_mid / (1 + (n0_mid - 1) / N_mid)
n_mid = int(np.ceil(n_mid_adj))
print(f"Midfielder population N={N_mid}; required sample size n = {n_mid}")

mid_sample = mid_pop.sample(n=n_mid, random_state=42)['Age']
print(f"Midfielder sample (n={len(mid_sample)}): mean={mid_sample.mean():.3f}, std={mid_sample.std(ddof=1):.3f}")

t_stat, p_val = stats.ttest_1samp(mid_sample, popmean=mu0)
print(f"\nOne-sample t-test: H0: mu_MID = {mu0:.3f} (defenders' mean age)")
print(f"t-statistic = {t_stat:.4f}, df = {len(mid_sample)-1}, p-value (two-tailed) = {p_val:.4f}")
alpha = 0.05
if p_val < alpha:
    print(f"Result: p < {alpha} -> Reject H0. Midfielders' mean age differs significantly from defenders'.")
else:
    print(f"Result: p >= {alpha} -> Fail to reject H0. No significant difference in mean age.")

# Save outputs for reuse
sample.to_csv('/home/claude/age_sample.csv', index=False)
mid_sample.to_frame().to_csv('/home/claude/mid_sample.csv', index=False)
df.to_csv('/home/claude/full_population_with_age.csv', index=False)
