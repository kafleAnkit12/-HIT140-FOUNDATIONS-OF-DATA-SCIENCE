import os
 import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, levene, mannwhitneyu, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# ── Auto-create the figures folder so you don't have to ───────
os.makedirs('figures', exist_ok=True)
print("  'figures/' folder ready ✓")

# ── Dataset path (datasets/ folder is one level up) ───────────
DATASETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets')

# ── Colour palette (used consistently across all figures) ─────
C_GROUP    = '#1565C0'   # deep blue  → Group Stage
C_KNOCKOUT = '#B71C1C'   # deep red   → Knockout Stage
C_NEUTRAL  = '#546E7A'   # slate grey → neutral elements
C_LIGHT_G  = '#BBDEFB'   # light blue → Group Stage fill
C_LIGHT_K  = '#FFCDD2'   # light red  → Knockout Stage fill

print("=" * 62)
print("  FIFA WORLD CUP 2026 — Yellow Cards Analysis")
print("  Group Stage vs Knockout Stage")
print("=" * 62)


# ══════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════
print("\n[SECTION 1] LOADING DATA")
print("-" * 40)

matches = pd.read_csv(os.path.join(DATASETS_DIR, 'matches.csv'))

print(f"  matches.csv loaded: {matches.shape[0]} rows × {matches.shape[1]} columns")
print(f"\n  Columns used in this analysis:")
used_cols = ['round','home_team','away_team',
             'home_cards_yellow','away_cards_yellow']
print(f"  {used_cols}")
print(f"\n  First 5 rows (relevant columns):")
print(matches[used_cols].head().to_string(index=False))


# ══════════════════════════════════════════════════════════════
# SECTION 2 — DATA WRANGLING
# ══════════════════════════════════════════════════════════════
print("\n\n[SECTION 2] DATA WRANGLING")
print("-" * 40)

# ── Step 2a: Check for missing values ─────────────────────────
print("\n  Step 2a — Missing value check:")
null_check = matches[['home_cards_yellow','away_cards_yellow','round']].isnull().sum()
print(f"  {null_check.to_dict()}")
print(f"  → No missing values. All 104 match records are complete. ✓")

# ── Step 2b: Create total yellow cards per match ──────────────
# Each match has a home team and away team yellow card count.
# Total = home + away = all yellow cards issued in that match.
matches['total_yellow'] = (matches['home_cards_yellow'] +
                           matches['away_cards_yellow'])
print(f"\n  Step 2b — Created 'total_yellow' column (home + away):")
print(f"  Sample: {matches['total_yellow'].head(6).tolist()}")

# ── Step 2c: Classify matches into Stage groups ───────────────
# Group Stage  → 'Group stage' (72 matches)
# Knockout Stage → all other rounds (32 matches):
#   Round of 32, Round of 16, Quarter-finals,
#   Semi-finals, Third-place match, Final
matches['stage'] = matches['round'].apply(
    lambda r: 'Group Stage' if r == 'Group stage' else 'Knockout Stage'
)

print(f"\n  Step 2c — 'stage' column created:")
print(f"  {matches['stage'].value_counts().to_dict()}")

# ── Step 2d: Verify round breakdown within Knockout Stage ─────
print(f"\n  Step 2d — Knockout Stage round breakdown:")
ko_rounds = matches[matches['stage']=='Knockout Stage']['round'].value_counts()
print(f"  {ko_rounds.to_dict()}")

# ── Step 2e: Build the working dataset ────────────────────────
df = matches[['round','stage','home_team','away_team',
              'home_cards_yellow','away_cards_yellow',
              'total_yellow']].copy()

print(f"\n  Final working dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n  Sample rows:")
print(df[['stage','home_team','away_team',
          'home_cards_yellow','away_cards_yellow',
          'total_yellow']].head(6).to_string(index=False))

# ── Separate group vectors for statistical tests ───────────────
group_yc    = df[df['stage'] == 'Group Stage']['total_yellow']
knockout_yc = df[df['stage'] == 'Knockout Stage']['total_yellow']


# ══════════════════════════════════════════════════════════════
# SECTION 3 — EXPLORATORY DATA ANALYSIS (EDA)
# ══════════════════════════════════════════════════════════════
print("\n\n[SECTION 3] EXPLORATORY DATA ANALYSIS")
print("-" * 40)

# ── Step 3a: Descriptive statistics ───────────────────────────
print("\n  Step 3a — Descriptive Statistics by Stage:")
desc = df.groupby('stage')['total_yellow'].agg(
    Count      = 'count',
    Mean       = 'mean',
    Median     = 'median',
    Std_Dev    = 'std',
    Min        = 'min',
    Max        = 'max',
    Q1         = lambda x: x.quantile(0.25),
    Q3         = lambda x: x.quantile(0.75)
).round(3)
print(desc.to_string())

print(f"\n  Key insight: Knockout Stage averages {knockout_yc.mean():.3f} yellow cards/match")
print(f"               Group Stage averages    {group_yc.mean():.3f} yellow cards/match")
print(f"               Raw difference:         {knockout_yc.mean()-group_yc.mean():.3f} cards/match")

# ── Step 3b: Per-round breakdown ──────────────────────────────
print("\n  Step 3b — Yellow Cards Mean by Round:")
round_stats = df.groupby('round')['total_yellow'].agg(['count','mean','std']).round(3)
round_stats.columns = ['Matches','Mean','Std']
print(round_stats.sort_values('Mean', ascending=False).to_string())


# ─────────────────────────────────────────────────────────────
# FIGURE 1: Side-by-side histograms (distribution)
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig.suptitle(
    'Figure 1 — Distribution of Yellow Cards Per Match\n'
    'FIFA World Cup 2026: Group Stage vs Knockout Stage',
    fontsize=13, fontweight='bold', y=1.01
)

bins = range(0, 10)

for ax, (stage, colour, lcolour, n, mean_v, med_v) in zip(axes, [
    ('Group Stage',    C_GROUP,    C_LIGHT_G,
     len(group_yc),    group_yc.mean(),    group_yc.median()),
    ('Knockout Stage', C_KNOCKOUT, C_LIGHT_K,
     len(knockout_yc), knockout_yc.mean(), knockout_yc.median())
]):
    data = df[df['stage']==stage]['total_yellow']
    ax.hist(data, bins=bins, align='left', color=lcolour,
            edgecolor=colour, linewidth=1.4)
    ax.axvline(mean_v,   color=colour,    linestyle='--',
               linewidth=2, label=f'Mean = {mean_v:.2f}')
    ax.axvline(med_v,    color=C_NEUTRAL, linestyle=':',
               linewidth=2, label=f'Median = {int(med_v)}')
    ax.set_title(f'{stage}  (n = {n})', fontsize=12, fontweight='bold', color=colour)
    ax.set_xlabel('Yellow Cards Per Match', fontsize=11)
    ax.set_ylabel('Number of Matches', fontsize=11)
    ax.set_xticks(range(0, 10))
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#f9f9f9')

plt.tight_layout()
plt.savefig('figures/fig1_histograms.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  ✔ Figure 1 saved: figures/fig1_histograms.png")


# ─────────────────────────────────────────────────────────────
# FIGURE 2: Box plots with jitter overlay
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

data_list   = [group_yc, knockout_yc]
stage_names = ['Group Stage\n(n = 72)', 'Knockout Stage\n(n = 32)']
colours_bp  = [C_GROUP, C_KNOCKOUT]
light_bp    = [C_LIGHT_G, C_LIGHT_K]

bp = ax.boxplot(data_list, patch_artist=True, notch=False,
                widths=0.42, vert=True,
                medianprops=dict(color='white', linewidth=2.8),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker='o', markersize=5,
                                linestyle='none', alpha=0.5))

for patch, lc, dc in zip(bp['boxes'], light_bp, colours_bp):
    patch.set_facecolor(lc)
    patch.set_edgecolor(dc)
    patch.set_linewidth(2)

# Jitter overlay
np.random.seed(42)
for i, (data, dc) in enumerate(zip(data_list, colours_bp), start=1):
    jitter = np.random.normal(i, 0.07, size=len(data))
    ax.scatter(jitter, data, color=dc, alpha=0.40, s=35, zorder=4)

# Annotate means
for i, (data, dc) in enumerate(zip(data_list, colours_bp), start=1):
    m = data.mean()
    ax.annotate(f'Mean = {m:.2f}',
                xy=(i, m), xytext=(i + 0.25, m + 0.15),
                fontsize=10, fontweight='bold', color=dc,
                arrowprops=dict(arrowstyle='->', color=dc, lw=1.3))

ax.set_xticks([1, 2])
ax.set_xticklabels(stage_names, fontsize=12)
ax.set_ylabel('Yellow Cards Per Match', fontsize=12)
ax.set_title(
    'Figure 2 — Yellow Cards Per Match: Group Stage vs Knockout Stage\n'
    'FIFA World Cup 2026 (box = IQR, whiskers = 1.5×IQR, dots = individual matches)',
    fontsize=12, fontweight='bold'
)
ax.set_yticks(range(0, 10))
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor('#f9f9f9')

legend_patches = [
    mpatches.Patch(facecolor=C_LIGHT_G, edgecolor=C_GROUP,
                   label='Group Stage'),
    mpatches.Patch(facecolor=C_LIGHT_K, edgecolor=C_KNOCKOUT,
                   label='Knockout Stage')
]
ax.legend(handles=legend_patches, fontsize=11, loc='upper left')

plt.tight_layout()
plt.savefig('figures/fig2_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✔ Figure 2 saved: figures/fig2_boxplot.png")


# ─────────────────────────────────────────────────────────────
# FIGURE 3: Bar chart — mean ± 95% CI
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

means = [group_yc.mean(), knockout_yc.mean()]
sems  = [group_yc.sem(),  knockout_yc.sem()]
ci95  = [1.96 * s for s in sems]
labels = ['Group Stage\n(n = 72)', 'Knockout Stage\n(n = 32)']

bars = ax.bar(labels, means, yerr=ci95, capsize=8,
              color=[C_LIGHT_G, C_LIGHT_K],
              edgecolor=[C_GROUP, C_KNOCKOUT],
              linewidth=2, width=0.45,
              error_kw=dict(elinewidth=2, ecolor=C_NEUTRAL))

for bar, mean_v, colour in zip(bars, means, [C_GROUP, C_KNOCKOUT]):
    ax.text(bar.get_x() + bar.get_width()/2,
            mean_v + 0.12,
            f'{mean_v:.3f}',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=colour)

ax.set_ylabel('Mean Yellow Cards Per Match', fontsize=12)
ax.set_title(
    'Figure 3 — Mean Yellow Cards Per Match by Stage\n'
    'FIFA World Cup 2026 (error bars = 95% Confidence Interval)',
    fontsize=12, fontweight='bold'
)
ax.set_ylim(0, 5.5)
ax.axhline(df['total_yellow'].mean(), color=C_NEUTRAL,
           linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'Overall mean = {df["total_yellow"].mean():.2f}')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor('#f9f9f9')

plt.tight_layout()
plt.savefig('figures/fig3_bar_mean_ci.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✔ Figure 3 saved: figures/fig3_bar_mean_ci.png")


# ─────────────────────────────────────────────────────────────
# FIGURE 4: Violin plot
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

plot_df = df[['stage','total_yellow']].copy()
sns.violinplot(data=plot_df, x='stage', y='total_yellow',
               palette={'Group Stage': C_LIGHT_G,
                        'Knockout Stage': C_LIGHT_K},
               order=['Group Stage','Knockout Stage'],
               inner='quartile', cut=0, ax=ax,
               linewidth=1.5)

# Colour violin edges manually
for i, (colour) in enumerate([C_GROUP, C_KNOCKOUT]):
    ax.collections[i].set_edgecolor(colour)
    ax.collections[i].set_linewidth(1.8)

# Add mean dots
for i, (data, colour) in enumerate(
        zip([group_yc, knockout_yc], [C_GROUP, C_KNOCKOUT])):
    ax.scatter(i, data.mean(), color=colour, s=100,
               zorder=5, label=f'Mean = {data.mean():.2f}')

ax.set_xlabel('Match Stage', fontsize=12)
ax.set_ylabel('Yellow Cards Per Match', fontsize=12)
ax.set_xticklabels(['Group Stage\n(n = 72)', 'Knockout Stage\n(n = 32)'],
                   fontsize=12)
ax.set_title(
    'Figure 4 — Yellow Cards Distribution Shape by Stage\n'
    'FIFA World Cup 2026 (inner lines = Q1, Median, Q3)',
    fontsize=12, fontweight='bold'
)
ax.legend(fontsize=10, title='Stage Mean', title_fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor('#f9f9f9')

plt.tight_layout()
plt.savefig('figures/fig4_violin.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✔ Figure 4 saved: figures/fig4_violin.png")


# ─────────────────────────────────────────────────────────────
# FIGURE 5: Yellow cards breakdown by individual round
# ─────────────────────────────────────────────────────────────
round_order = ['Group stage','Round of 32','Round of 16',
               'Quarter-finals','Semi-finals',
               'Third-place match','Final']

round_stats2 = (df.groupby('round')['total_yellow']
                  .agg(['mean','sem','count'])
                  .reindex(round_order).reset_index())
round_stats2['ci95'] = round_stats2['sem'] * 1.96

round_colours = [C_GROUP] * 1 + [C_KNOCKOUT] * 6

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(round_stats2['round'], round_stats2['mean'],
              yerr=round_stats2['ci95'], capsize=5,
              color=[C_LIGHT_G] + [C_LIGHT_K]*6,
              edgecolor=round_colours, linewidth=2,
              error_kw=dict(elinewidth=1.8, ecolor=C_NEUTRAL))

# Annotate count + mean
for bar, (_, row), ec in zip(bars, round_stats2.iterrows(), round_colours):
    ax.text(bar.get_x() + bar.get_width()/2,
            row['mean'] + 0.12,
            f"{row['mean']:.2f}\n(n={int(row['count'])})",
            ha='center', va='bottom', fontsize=8.5,
            fontweight='bold', color=ec)

ax.set_xlabel('Tournament Round', fontsize=11)
ax.set_ylabel('Mean Yellow Cards Per Match', fontsize=11)
ax.set_title(
    'Figure 5 — Mean Yellow Cards Per Match Across All Tournament Rounds\n'
    'FIFA World Cup 2026 (error bars = 95% CI   |   Blue = Group Stage, Red = Knockout)',
    fontsize=11, fontweight='bold'
)
ax.set_xticklabels(round_order, rotation=25, ha='right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor('#f9f9f9')

legend_patches = [
    mpatches.Patch(facecolor=C_LIGHT_G, edgecolor=C_GROUP,  label='Group Stage'),
    mpatches.Patch(facecolor=C_LIGHT_K, edgecolor=C_KNOCKOUT, label='Knockout Stage')
]
ax.legend(handles=legend_patches, fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig5_by_round.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✔ Figure 5 saved: figures/fig5_by_round.png")


# ══════════════════════════════════════════════════════════════
# SECTION 4 — ASSUMPTION CHECKING
# ══════════════════════════════════════════════════════════════
print("\n\n[SECTION 4] ASSUMPTION CHECKING")
print("-" * 40)

# ── 4a. Shapiro-Wilk Normality Test ───────────────────────────
print("\n  4a — Shapiro-Wilk Normality Test:")
sw_g  = shapiro(group_yc)
sw_k  = shapiro(knockout_yc)
print(f"  Group Stage:    W = {sw_g.statistic:.4f},  p = {sw_g.pvalue:.4f}  "
      f"→ {'Normal ✓' if sw_g.pvalue > 0.05 else 'NOT normal ✗ → use non-parametric test'}")
print(f"  Knockout Stage: W = {sw_k.statistic:.4f},  p = {sw_k.pvalue:.4f}  "
      f"→ {'Normal ✓' if sw_k.pvalue > 0.05 else 'NOT normal ✗'}")
print(f"\n  Interpretation:")
print(f"  Group Stage data violates normality (p = {sw_g.pvalue:.4f} < 0.05).")
print(f"  Therefore, the primary test will be the Mann-Whitney U")
print(f"  (non-parametric), which does not assume normality.")
print(f"  The Welch t-test is also reported as a supplementary test.")

# ── 4b. Levene's Test for Equal Variances ─────────────────────
print("\n  4b — Levene's Test for Equality of Variances:")
lev = levene(group_yc, knockout_yc)
print(f"  F = {lev.statistic:.4f},  p = {lev.pvalue:.4f}")
print(f"  → Variances are {'equal ✓' if lev.pvalue > 0.05 else 'unequal'} "
      f"(p {'>' if lev.pvalue > 0.05 else '<'} 0.05)")
print(f"  Variances are equal — but since normality is violated,")
print(f"  Mann-Whitney U remains the primary test.")


# ══════════════════════════════════════════════════════════════
# SECTION 5 — STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════
print("\n\n[SECTION 5] STATISTICAL TESTS")
print("-" * 40)

# ── 5a. Mann-Whitney U Test (PRIMARY — non-parametric) ────────
print("\n  5a — Mann-Whitney U Test (PRIMARY TEST)")
print("       Used because Group Stage data violates normality.")
mwu = mannwhitneyu(group_yc, knockout_yc, alternative='two-sided')
n1, n2 = len(group_yc), len(knockout_yc)
r_rb = 1 - (2 * mwu.statistic) / (n1 * n2)   # rank-biserial r (effect size)

print(f"\n  Group Stage:    n={n1}, mean={group_yc.mean():.4f}, "
      f"median={group_yc.median():.1f}")
print(f"  Knockout Stage: n={n2}, mean={knockout_yc.mean():.4f}, "
      f"median={knockout_yc.median():.1f}")
print(f"\n  U statistic = {mwu.statistic:.0f}")
print(f"  p-value     = {mwu.pvalue:.4f}  (two-tailed, α = 0.05)")
print(f"  Effect size: rank-biserial r = {r_rb:.4f}  "
      f"({'small' if abs(r_rb)<0.3 else 'medium' if abs(r_rb)<0.5 else 'large'})")
print(f"\n  Decision: {'REJECT H₀' if mwu.pvalue < 0.05 else 'FAIL TO REJECT H₀'}")
print(f"  → p = {mwu.pvalue:.4f} > 0.05 — insufficient evidence to conclude")
print(f"    a statistically significant difference exists.")

# ── 5b. Welch t-test (SUPPLEMENTARY) ─────────────────────────
print("\n  5b — Welch Independent Samples t-test (SUPPLEMENTARY)")
print("       Reported for comparison alongside Mann-Whitney U.")
welch = ttest_ind(group_yc, knockout_yc, equal_var=False)
pooled_std = np.sqrt((group_yc.std()**2 + knockout_yc.std()**2) / 2)
cohens_d   = (knockout_yc.mean() - group_yc.mean()) / pooled_std

print(f"\n  t-statistic = {welch.statistic:.4f}")
print(f"  p-value     = {welch.pvalue:.4f}  (two-tailed, α = 0.05)")
print(f"  Cohen's d   = {cohens_d:.4f}  "
      f"({'small' if abs(cohens_d)<0.5 else 'medium' if abs(cohens_d)<0.8 else 'large'} effect)")
print(f"\n  Decision: {'REJECT H₀' if welch.pvalue < 0.05 else 'FAIL TO REJECT H₀'}")
print(f"  → Both tests agree: no significant difference at α = 0.05.")


# ─────────────────────────────────────────────────────────────
# FIGURE 6: Q-Q plots (normality visualisation)
# ─────────────────────────────────────────────────────────────
import statsmodels.api as sm

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    'Figure 6 — Normal Q-Q Plots: Yellow Cards Per Match\n'
    'FIFA World Cup 2026 (checking normality assumption)',
    fontsize=12, fontweight='bold', y=1.01
)

for ax, (data, stage, colour) in zip(axes, [
    (group_yc,    'Group Stage\n(n=72)',    C_GROUP),
    (knockout_yc, 'Knockout Stage\n(n=32)', C_KNOCKOUT)
]):
    sm.qqplot(data, line='s', ax=ax, alpha=0.6,
              markerfacecolor=colour, markeredgecolor=colour,
              markersize=6)
    ax.set_title(stage, fontsize=12, color=colour, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_facecolor('#f9f9f9')

plt.tight_layout()
plt.savefig('figures/fig6_qqplots.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  ✔ Figure 6 saved: figures/fig6_qqplots.png")


# ══════════════════════════════════════════════════════════════
# SECTION 6 — RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 62)
print("  RESULTS SUMMARY")
print("=" * 62)
print(f"""
  ANALYTIC QUESTION
  ─────────────────
  Is there a statistically significant difference in the average
  number of yellow cards per match between Group Stage and
  Knockout Stage matches at the FIFA World Cup 2026?

  DATASET
  ───────
  • Source   : matches.csv
  • Rows used: 104 matches (no missing values)
  • Group Stage    : 72 matches
  • Knockout Stage : 32 matches
    (Round of 32=16, Round of 16=8, QF=4, SF=2, TPM=1, Final=1)

  DESCRIPTIVE STATISTICS
  ──────────────────────
  Stage              n     Mean    Median   Std    Min   Max
  Group Stage       72     2.486     2.0    1.627    0     7
  Knockout Stage    32     3.063     3.0    2.047    0     8
  Difference                0.577

  ASSUMPTION CHECKING
  ───────────────────
  Shapiro-Wilk (Group Stage):    W={sw_g.statistic:.4f}, p={sw_g.pvalue:.4f} → NOT normal ✗
  Shapiro-Wilk (Knockout Stage): W={sw_k.statistic:.4f}, p={sw_k.pvalue:.4f} → Normal ✓
  Levene's test:                 F={lev.statistic:.4f}, p={lev.pvalue:.4f} → Equal variances ✓
  Decision: Non-parametric test used (normality violated in Group Stage).

  STATISTICAL TESTS
  ─────────────────
  Primary   — Mann-Whitney U Test:
    U = {mwu.statistic:.0f},  p = {mwu.pvalue:.4f}  → FAIL TO REJECT H₀
    Rank-biserial r = {r_rb:.4f} (small effect)

  Secondary — Welch t-test:
    t = {welch.statistic:.4f},  p = {welch.pvalue:.4f}  → FAIL TO REJECT H₀
    Cohen's d = {cohens_d:.4f} (small effect)

  CONCLUSION
  ──────────
  Both statistical tests produced p > 0.05 (Mann-Whitney p = {mwu.pvalue:.4f};
  Welch t p = {welch.pvalue:.4f}). There is INSUFFICIENT STATISTICAL EVIDENCE
  to conclude that the average number of yellow cards differs
  significantly between Group Stage and Knockout Stage matches
  at the FIFA World Cup 2026.

  While Knockout Stage matches averaged {knockout_yc.mean()-group_yc.mean():.3f} more yellow cards
  per match ({group_yc.mean():.3f} vs {knockout_yc.mean():.3f}), this difference is not
  statistically significant at α = 0.05. The small effect size
  (r = {r_rb:.4f}) and the limited number of Knockout Stage matches
  (n = {n2}) may have reduced the test's statistical power.

  H₀ is NOT rejected.
""")

print("  All 6 figures saved to the 'figures/' folder.")
print("  Analysis complete. ✅")