"""
Feature Correlation & Conflict Analysis

Analyzes the 64 strategy features to identify:
1. Conflicting pairs (strong negative correlation) 
2. Redundant pairs (very high positive correlation)
3. Feature groups for Gated architecture design

Author: Golden Breeze Team
Date: 2025-12-05
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from aimodule.data_pipeline.strategy_signals import StrategySignalsGenerator


def load_strategy_features(npz_path: str) -> np.ndarray:
    """Load strategy features from dataset."""
    print(f"📂 Loading dataset from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    x_strategy = data['x_strategy']
    print(f"   Strategy features shape: {x_strategy.shape}")
    return x_strategy


def get_feature_names() -> list:
    """Get strategy feature names from generator."""
    gen = StrategySignalsGenerator()
    # Generate dummy data to get column names
    dummy_df = pd.DataFrame({
        'open': [1.0] * 100,
        'high': [1.1] * 100,
        'low': [0.9] * 100,
        'close': [1.0] * 100,
        'tick_volume': [1000] * 100,
    })
    signals = gen.generate_all_signals(dummy_df)
    return list(signals.columns)


def compute_correlation_matrix(features: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Compute Spearman correlation matrix."""
    print("\n📊 Computing Spearman correlation matrix...")
    
    # Sample if too large
    n_samples = len(features)
    if n_samples > 50000:
        print(f"   Sampling {50000:,} of {n_samples:,} for efficiency...")
        idx = np.random.choice(n_samples, 50000, replace=False)
        features = features[idx]
    
    # Compute correlation
    n_features = features.shape[1]
    corr_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(i, n_features):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # Spearman correlation
                rho, _ = stats.spearmanr(features[:, i], features[:, j])
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho
        
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{n_features} features...")
    
    # Create DataFrame
    df_corr = pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)
    return df_corr


def find_conflicts(corr_matrix: pd.DataFrame, threshold: float = -0.3) -> list:
    """Find strongly conflicting feature pairs (negative correlation)."""
    conflicts = []
    features = corr_matrix.columns.tolist()
    
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            corr = corr_matrix.iloc[i, j]
            if corr < threshold:
                conflicts.append({
                    'feature_a': features[i],
                    'feature_b': features[j],
                    'correlation': corr,
                    'conflict_strength': abs(corr)
                })
    
    # Sort by conflict strength
    conflicts.sort(key=lambda x: x['conflict_strength'], reverse=True)
    return conflicts


def find_redundant(corr_matrix: pd.DataFrame, threshold: float = 0.90) -> list:
    """Find highly redundant feature pairs (very high positive correlation)."""
    redundant = []
    features = corr_matrix.columns.tolist()
    
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            corr = corr_matrix.iloc[i, j]
            if corr > threshold:
                redundant.append({
                    'feature_a': features[i],
                    'feature_b': features[j],
                    'correlation': corr,
                })
    
    # Sort by correlation
    redundant.sort(key=lambda x: x['correlation'], reverse=True)
    return redundant


def categorize_features(feature_names: list) -> dict:
    """Categorize features into logical groups based on names."""
    categories = {
        'trend': [],
        'oscillator': [],
        'momentum': [],
        'volatility': [],
        'volume': [],
        'pattern': [],
        'regime': [],
        'other': []
    }
    
    for name in feature_names:
        name_lower = name.lower()
        
        if any(x in name_lower for x in ['trend', 'ema', 'sma', 'ma_', 'direction', 'slope']):
            categories['trend'].append(name)
        elif any(x in name_lower for x in ['rsi', 'stoch', 'cci', 'williams', 'osc']):
            categories['oscillator'].append(name)
        elif any(x in name_lower for x in ['momentum', 'roc', 'velocity', 'accel']):
            categories['momentum'].append(name)
        elif any(x in name_lower for x in ['atr', 'volatility', 'bb_', 'stddev', 'range']):
            categories['volatility'].append(name)
        elif any(x in name_lower for x in ['volume', 'vol_', 'obv', 'mfi', 'vwap']):
            categories['volume'].append(name)
        elif any(x in name_lower for x in ['pattern', 'candle', 'doji', 'hammer', 'engulf']):
            categories['pattern'].append(name)
        elif any(x in name_lower for x in ['regime', 'state', 'phase', 'market']):
            categories['regime'].append(name)
        else:
            categories['other'].append(name)
    
    return categories


def build_conflict_groups(conflicts: list, threshold_count: int = 2) -> dict:
    """Group features by how often they conflict with each other."""
    conflict_count = defaultdict(int)
    conflict_partners = defaultdict(list)
    
    for c in conflicts:
        conflict_count[c['feature_a']] += 1
        conflict_count[c['feature_b']] += 1
        conflict_partners[c['feature_a']].append((c['feature_b'], c['correlation']))
        conflict_partners[c['feature_b']].append((c['feature_a'], c['correlation']))
    
    # Features that conflict with many others
    high_conflict = {k: v for k, v in conflict_count.items() if v >= threshold_count}
    high_conflict = dict(sorted(high_conflict.items(), key=lambda x: x[1], reverse=True))
    
    return high_conflict, conflict_partners


def build_redundancy_groups(redundant: list) -> list:
    """Build groups of redundant features using union-find."""
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Union redundant pairs
    for r in redundant:
        union(r['feature_a'], r['feature_b'])
    
    # Build groups
    groups = defaultdict(list)
    for feat in parent.keys():
        root = find(feat)
        groups[root].append(feat)
    
    # Filter groups with >1 member and sort by size
    groups = {k: v for k, v in groups.items() if len(v) > 1}
    groups = dict(sorted(groups.items(), key=lambda x: len(x[1]), reverse=True))
    
    return groups


def generate_report(
    corr_matrix: pd.DataFrame,
    conflicts: list,
    redundant: list,
    categories: dict,
    high_conflict: dict,
    redundancy_groups: dict,
    output_path: str
):
    """Generate comprehensive analysis report."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("FEATURE CORRELATION & CONFLICT ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: 2025-12-05")
    lines.append(f"Dataset: v4_6year_dataset.npz")
    lines.append(f"Total features analyzed: {len(corr_matrix)}")
    lines.append("")
    
    # ==== SECTION 1: CATEGORY BREAKDOWN ====
    lines.append("=" * 80)
    lines.append("SECTION 1: FEATURE CATEGORIES")
    lines.append("=" * 80)
    lines.append("")
    
    for cat, features in categories.items():
        if features:
            lines.append(f"📁 {cat.upper()} ({len(features)} features):")
            for f in features:
                lines.append(f"   - {f}")
            lines.append("")
    
    # ==== SECTION 2: TOP CONFLICTS ====
    lines.append("=" * 80)
    lines.append("SECTION 2: TOP CONFLICTING PAIRS (Negative Correlation)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("These features tend to give OPPOSITE signals:")
    lines.append("(When one says BUY, the other says SELL)")
    lines.append("")
    
    for i, c in enumerate(conflicts[:30], 1):
        lines.append(f"{i:2d}. {c['feature_a']:30s} ↔ {c['feature_b']:30s} | r = {c['correlation']:+.3f}")
    
    if len(conflicts) > 30:
        lines.append(f"\n... and {len(conflicts) - 30} more conflicting pairs")
    
    lines.append("")
    
    # ==== SECTION 3: MOST CONFLICTING FEATURES ====
    lines.append("=" * 80)
    lines.append("SECTION 3: MOST CONFLICTING FEATURES")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Features that conflict with MANY other features:")
    lines.append("(These are 'controversial' - often disagree with others)")
    lines.append("")
    
    for feat, count in list(high_conflict.items())[:20]:
        cat = "?"
        for c, feats in categories.items():
            if feat in feats:
                cat = c[:3].upper()
                break
        lines.append(f"   {feat:40s} | conflicts with {count:2d} features [{cat}]")
    
    lines.append("")
    
    # ==== SECTION 4: REDUNDANT PAIRS ====
    lines.append("=" * 80)
    lines.append("SECTION 4: TOP REDUNDANT PAIRS (High Positive Correlation)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("These features are nearly IDENTICAL (duplicates):")
    lines.append("(Can potentially be merged or one gated by another)")
    lines.append("")
    
    for i, r in enumerate(redundant[:30], 1):
        lines.append(f"{i:2d}. {r['feature_a']:30s} ≈ {r['feature_b']:30s} | r = {r['correlation']:.3f}")
    
    if len(redundant) > 30:
        lines.append(f"\n... and {len(redundant) - 30} more redundant pairs")
    
    lines.append("")
    
    # ==== SECTION 5: REDUNDANCY GROUPS ====
    lines.append("=" * 80)
    lines.append("SECTION 5: REDUNDANCY GROUPS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Features that are redundant with each other (can be grouped):")
    lines.append("")
    
    for i, (root, members) in enumerate(redundancy_groups.items(), 1):
        lines.append(f"GROUP {i} ({len(members)} features):")
        for m in members:
            lines.append(f"   - {m}")
        lines.append("")
    
    # ==== SECTION 6: GATING ARCHITECTURE RECOMMENDATIONS ====
    lines.append("=" * 80)
    lines.append("SECTION 6: GATING ARCHITECTURE RECOMMENDATIONS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Based on conflict analysis, we recommend grouping features as:")
    lines.append("")
    
    # Suggest gating groups
    lines.append("🔹 GATE 1: TREND FOLLOWERS")
    lines.append("   Purpose: Long-term direction signals")
    lines.append("   When active: Clear trending market")
    if categories['trend']:
        lines.append("   Features: " + ", ".join(categories['trend'][:5]) + "...")
    lines.append("")
    
    lines.append("🔹 GATE 2: OSCILLATORS (MEAN-REVERSION)")
    lines.append("   Purpose: Overbought/Oversold signals")
    lines.append("   When active: Ranging/Sideways market")
    if categories['oscillator']:
        lines.append("   Features: " + ", ".join(categories['oscillator'][:5]) + "...")
    lines.append("")
    
    lines.append("🔹 GATE 3: MOMENTUM")
    lines.append("   Purpose: Strength of current move")
    lines.append("   When active: Always (weight by regime)")
    if categories['momentum']:
        lines.append("   Features: " + ", ".join(categories['momentum'][:5]) + "...")
    lines.append("")
    
    lines.append("🔹 GATE 4: VOLATILITY CONTEXT")
    lines.append("   Purpose: Position sizing & stop placement")
    lines.append("   When active: Always (context layer)")
    if categories['volatility']:
        lines.append("   Features: " + ", ".join(categories['volatility'][:5]) + "...")
    lines.append("")
    
    lines.append("🔹 GATE 5: VOLUME CONFIRMATION")
    lines.append("   Purpose: Validate price moves with volume")
    lines.append("   When active: Always (confirmation layer)")
    if categories['volume']:
        lines.append("   Features: " + ", ".join(categories['volume'][:5]) + "...")
    lines.append("")
    
    # ==== SECTION 7: KEY INSIGHT ====
    lines.append("=" * 80)
    lines.append("SECTION 7: KEY INSIGHTS FOR MODEL ARCHITECTURE")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. CONFLICT RESOLUTION:")
    lines.append("   - Trend vs Oscillator is the main conflict")
    lines.append("   - Solution: Use REGIME DETECTOR to gate which group is active")
    lines.append("   - In TREND regime: weight Trend features higher")
    lines.append("   - In RANGE regime: weight Oscillator features higher")
    lines.append("")
    lines.append("2. REDUNDANCY HANDLING:")
    lines.append("   - Don't drop redundant features!")
    lines.append("   - Instead: Use attention mechanism to let model learn which to focus on")
    lines.append("   - Or: Average within redundancy groups before feeding to LSTM")
    lines.append("")
    lines.append("3. RECOMMENDED ARCHITECTURE:")
    lines.append("   ┌─────────────┐")
    lines.append("   │ Raw Features│ (64)")
    lines.append("   └──────┬──────┘")
    lines.append("          │")
    lines.append("   ┌──────▼──────┐")
    lines.append("   │ Group by    │")
    lines.append("   │ Category    │")
    lines.append("   └──────┬──────┘")
    lines.append("          │")
    lines.append("   ┌──────▼──────┐     ┌────────────┐")
    lines.append("   │ Regime      │◄────│ Volatility │")
    lines.append("   │ Detector    │     │ Context    │")
    lines.append("   └──────┬──────┘     └────────────┘")
    lines.append("          │")
    lines.append("   ┌──────▼──────┐")
    lines.append("   │ Gated       │")
    lines.append("   │ Feature Mix │")
    lines.append("   └──────┬──────┘")
    lines.append("          │")
    lines.append("   ┌──────▼──────┐")
    lines.append("   │ LSTM Head   │")
    lines.append("   └─────────────┘")
    lines.append("")
    
    # ==== STATISTICS SUMMARY ====
    lines.append("=" * 80)
    lines.append("STATISTICS SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total features: {len(corr_matrix)}")
    lines.append(f"Conflicting pairs (r < -0.3): {len(conflicts)}")
    lines.append(f"Highly redundant pairs (r > 0.9): {len(redundant)}")
    lines.append(f"Redundancy groups: {len(redundancy_groups)}")
    lines.append(f"High-conflict features: {len(high_conflict)}")
    lines.append("")
    
    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\n✅ Report saved to: {output_path}")
    
    return '\n'.join(lines)


def main():
    print("=" * 60)
    print("FEATURE CORRELATION & CONFLICT ANALYSIS")
    print("=" * 60)
    
    # Load data
    npz_path = "data/prepared/v4_6year_dataset.npz"
    features = load_strategy_features(npz_path)
    
    # Get feature names
    print("\n📋 Getting feature names...")
    try:
        feature_names = get_feature_names()
        if len(feature_names) != features.shape[1]:
            print(f"   Warning: Names ({len(feature_names)}) != Features ({features.shape[1]})")
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]
    except Exception as e:
        print(f"   Error getting names: {e}")
        feature_names = [f"feature_{i}" for i in range(features.shape[1])]
    
    print(f"   Feature names: {len(feature_names)}")
    
    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(features, feature_names)
    
    # Save correlation matrix
    corr_path = "logs/correlation_matrix.csv"
    Path(corr_path).parent.mkdir(parents=True, exist_ok=True)
    corr_matrix.to_csv(corr_path)
    print(f"   Correlation matrix saved to: {corr_path}")
    
    # Find conflicts and redundancies
    print("\n🔍 Analyzing conflicts and redundancies...")
    conflicts = find_conflicts(corr_matrix, threshold=-0.3)
    redundant = find_redundant(corr_matrix, threshold=0.90)
    
    print(f"   Conflicting pairs: {len(conflicts)}")
    print(f"   Redundant pairs: {len(redundant)}")
    
    # Categorize features
    categories = categorize_features(feature_names)
    
    # Build conflict groups
    high_conflict, conflict_partners = build_conflict_groups(conflicts)
    
    # Build redundancy groups
    redundancy_groups = build_redundancy_groups(redundant)
    
    # Generate report
    report = generate_report(
        corr_matrix=corr_matrix,
        conflicts=conflicts,
        redundant=redundant,
        categories=categories,
        high_conflict=high_conflict,
        redundancy_groups=redundancy_groups,
        output_path="logs/feature_conflicts.txt"
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    print(f"Total features: {len(feature_names)}")
    print(f"Conflicting pairs: {len(conflicts)}")
    print(f"Redundant pairs: {len(redundant)}")
    print(f"Redundancy groups: {len(redundancy_groups)}")
    
    if conflicts:
        print("\nTop 5 Conflicts:")
        for c in conflicts[:5]:
            print(f"  {c['feature_a']} ↔ {c['feature_b']} (r={c['correlation']:.3f})")
    
    if redundant:
        print("\nTop 5 Redundant:")
        for r in redundant[:5]:
            print(f"  {r['feature_a']} ≈ {r['feature_b']} (r={r['correlation']:.3f})")
    
    print("\n✅ Analysis complete! See logs/feature_conflicts.txt for full report")


if __name__ == "__main__":
    main()
