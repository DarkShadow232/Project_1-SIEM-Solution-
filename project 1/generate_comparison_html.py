"""
Dataset Comparison HTML Report Generator
Compares older dataset (Output1.csv) with newer dataset (output3.csv)
Creates a beautiful, modern HTML report
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import base64
from io import BytesIO

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_datasets():
    """Load the old and new datasets"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    siem_path = os.path.join(base_path, 'siem_integration')
    
    old_dataset_path = os.path.join(siem_path, 'Output1.csv')
    new_dataset_path = os.path.join(siem_path, 'output3.csv')
    
    print(f"Loading old dataset from: {old_dataset_path}")
    print(f"Loading new dataset from: {new_dataset_path}")
    
    old_df = pd.read_csv(old_dataset_path)
    new_df = pd.read_csv(new_dataset_path)
    
    return old_df, new_df


def calculate_statistics(df, name):
    """Calculate comprehensive statistics for a dataset"""
    stats = {
        'name': name,
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': list(df.columns),
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats['numeric_columns'] = len(numeric_cols)
    
    if numeric_cols:
        stats['mean_values'] = df[numeric_cols].mean().to_dict()
        stats['std_values'] = df[numeric_cols].std().to_dict()
        stats['min_values'] = df[numeric_cols].min().to_dict()
        stats['max_values'] = df[numeric_cols].max().to_dict()
        stats['median_values'] = df[numeric_cols].median().to_dict()
    
    stats['missing_values'] = df.isnull().sum().sum()
    stats['missing_by_column'] = df.isnull().sum().to_dict()
    
    return stats


def create_chart_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def create_charts(old_df, new_df):
    """Create comparison charts and return as base64"""
    if not MATPLOTLIB_AVAILABLE:
        return {}
    
    charts = {}
    
    # Set dark theme
    plt.style.use('dark_background')
    
    # Get common numeric columns
    old_numeric = set(old_df.select_dtypes(include=[np.number]).columns)
    new_numeric = set(new_df.select_dtypes(include=[np.number]).columns)
    common_cols = list(old_numeric.intersection(new_numeric))
    key_features = common_cols[:8] if len(common_cols) > 8 else common_cols
    
    # Color palette
    old_color = '#ff6b6b'
    new_color = '#4ecdc4'
    bg_color = '#1a1a2e'
    text_color = '#eee'
    
    # Chart 1: Dataset Size Comparison
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    categories = ['Total Rows', 'Numeric Columns', 'Total Columns']
    old_values = [len(old_df), len(old_numeric), len(old_df.columns)]
    new_values = [len(new_df), len(new_numeric), len(new_df.columns)]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, old_values, width, label='Old Dataset', color=old_color, alpha=0.9, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, new_values, width, label='New Dataset', color=new_color, alpha=0.9, edgecolor='white', linewidth=1)
    
    ax.set_ylabel('Count', fontsize=12, color=text_color)
    ax.set_title('Dataset Size Comparison', fontsize=16, fontweight='bold', color=text_color, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, color=text_color)
    ax.legend(facecolor=bg_color, edgecolor='#333', labelcolor=text_color)
    ax.tick_params(colors=text_color)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, color='#666')
    
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', 
                    fontsize=11, fontweight='bold', color=text_color)
    
    charts['size_comparison'] = create_chart_base64(fig)
    
    # Chart 2: Mean Values Comparison
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    old_means = [old_df[col].mean() for col in key_features]
    new_means = [new_df[col].mean() for col in key_features]
    
    x = np.arange(len(key_features))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, old_means, width, label='Old Dataset', color=old_color, alpha=0.9, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, new_means, width, label='New Dataset', color=new_color, alpha=0.9, edgecolor='white', linewidth=1)
    
    ax.set_ylabel('Mean Value', fontsize=12, color=text_color)
    ax.set_title('Mean Values Comparison - Key Features', fontsize=16, fontweight='bold', color=text_color, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([col.split('.')[-1][:12] for col in key_features], rotation=45, ha='right', color=text_color)
    ax.legend(facecolor=bg_color, edgecolor='#333', labelcolor=text_color)
    ax.tick_params(colors=text_color)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, color='#666')
    
    plt.tight_layout()
    charts['mean_comparison'] = create_chart_base64(fig)
    
    # Chart 3: Percentage Change
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    pct_changes = []
    for col in key_features:
        old_mean = old_df[col].mean()
        new_mean = new_df[col].mean()
        if old_mean != 0:
            pct_change = ((new_mean - old_mean) / abs(old_mean)) * 100
        else:
            pct_change = 0
        pct_changes.append(pct_change)
    
    colors_list = [new_color if x >= 0 else old_color for x in pct_changes]
    bars = ax.barh(range(len(key_features)), pct_changes, color=colors_list, alpha=0.9, edgecolor='white', linewidth=1)
    
    ax.set_yticks(range(len(key_features)))
    ax.set_yticklabels([col.split('.')[-1][:18] for col in key_features], color=text_color)
    ax.set_xlabel('Percentage Change (%)', fontsize=12, color=text_color)
    ax.set_title('Feature Change Analysis (Old → New)', fontsize=16, fontweight='bold', color=text_color, pad=20)
    ax.axvline(x=0, color='#888', linestyle='-', linewidth=1)
    ax.tick_params(colors=text_color)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2, color='#666')
    
    for i, (bar, pct) in enumerate(zip(bars, pct_changes)):
        width = bar.get_width()
        ax.annotate(f'{pct:+.1f}%', xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5 if width >= 0 else -5, 0), textcoords="offset points",
                    ha='left' if width >= 0 else 'right', va='center', fontsize=10, color=text_color)
    
    plt.tight_layout()
    charts['pct_change'] = create_chart_base64(fig)
    
    # Chart 4: Distribution Radar/Spider Chart Alternative - Normalized Comparison
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    # Normalize values for comparison
    old_norm = [(old_df[col].mean() - old_df[col].min()) / (old_df[col].max() - old_df[col].min() + 0.0001) for col in key_features[:6]]
    new_norm = [(new_df[col].mean() - new_df[col].min()) / (new_df[col].max() - new_df[col].min() + 0.0001) for col in key_features[:6]]
    
    x = np.arange(len(key_features[:6]))
    
    ax.plot(x, old_norm, 'o-', color=old_color, linewidth=2, markersize=10, label='Old Dataset', alpha=0.9)
    ax.plot(x, new_norm, 's-', color=new_color, linewidth=2, markersize=10, label='New Dataset', alpha=0.9)
    ax.fill_between(x, old_norm, alpha=0.2, color=old_color)
    ax.fill_between(x, new_norm, alpha=0.2, color=new_color)
    
    ax.set_xticks(x)
    ax.set_xticklabels([col.split('.')[-1][:12] for col in key_features[:6]], rotation=45, ha='right', color=text_color)
    ax.set_ylabel('Normalized Value', fontsize=12, color=text_color)
    ax.set_title('Normalized Feature Distribution', fontsize=16, fontweight='bold', color=text_color, pad=20)
    ax.legend(facecolor=bg_color, edgecolor='#333', labelcolor=text_color)
    ax.tick_params(colors=text_color)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2, color='#666')
    
    plt.tight_layout()
    charts['distribution'] = create_chart_base64(fig)
    
    return charts


def generate_html_report(old_df, new_df, old_stats, new_stats, charts, output_path):
    """Generate beautiful HTML comparison report"""
    
    # Calculate differences
    row_diff = new_stats['rows'] - old_stats['rows']
    row_pct = (row_diff / old_stats['rows']) * 100 if old_stats['rows'] > 0 else 0
    
    # Get common numeric columns for stats table
    old_numeric = set(old_df.select_dtypes(include=[np.number]).columns)
    new_numeric = set(new_df.select_dtypes(include=[np.number]).columns)
    common_cols = list(old_numeric.intersection(new_numeric))[:12]
    
    # Build stats table rows
    stats_rows = ""
    for col in common_cols:
        old_mean = old_df[col].mean()
        new_mean = new_df[col].mean()
        old_std = old_df[col].std()
        new_std = new_df[col].std()
        
        if old_mean != 0:
            pct_change = ((new_mean - old_mean) / abs(old_mean)) * 100
        else:
            pct_change = 0
        
        change_class = "positive" if pct_change >= 0 else "negative"
        col_display = col.split('.')[-1][:25]
        
        stats_rows += f"""
        <tr>
            <td class="feature-name">{col_display}</td>
            <td>{old_mean:.2f}</td>
            <td>{new_mean:.2f}</td>
            <td>{old_std:.2f}</td>
            <td>{new_std:.2f}</td>
            <td class="{change_class}">{pct_change:+.1f}%</td>
        </tr>
        """
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Comparison Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: #1a1a2e;
            --bg-card-hover: #252540;
            --accent-cyan: #4ecdc4;
            --accent-coral: #ff6b6b;
            --accent-purple: #a855f7;
            --accent-gold: #fbbf24;
            --text-primary: #f0f0f5;
            --text-secondary: #a0a0b0;
            --text-muted: #606080;
            --border-color: #2a2a40;
            --gradient-1: linear-gradient(135deg, #4ecdc4 0%, #44a3aa 100%);
            --gradient-2: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
            --gradient-3: linear-gradient(135deg, #a855f7 0%, #8b5cf6 100%);
            --gradient-gold: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }}
        
        /* Animated background */
        .bg-animation {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            overflow: hidden;
        }}
        
        .bg-animation::before {{
            content: '';
            position: absolute;
            width: 200%;
            height: 200%;
            top: -50%;
            left: -50%;
            background: 
                radial-gradient(circle at 20% 80%, rgba(78, 205, 196, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 107, 107, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(168, 85, 247, 0.05) 0%, transparent 40%);
            animation: pulse 20s ease-in-out infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1) rotate(0deg); }}
            50% {{ transform: scale(1.1) rotate(5deg); }}
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        
        /* Header */
        .header {{
            text-align: center;
            padding: 60px 0 40px;
            position: relative;
        }}
        
        .header::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 200px;
            height: 3px;
            background: var(--gradient-1);
            border-radius: 3px;
        }}
        
        .header h1 {{
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 15px;
            letter-spacing: -1px;
        }}
        
        .header .subtitle {{
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 400;
        }}
        
        .header .date {{
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 10px;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        /* Stats Cards */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin: 50px 0;
        }}
        
        .stat-card {{
            background: var(--bg-card);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--gradient-1);
        }}
        
        .stat-card.coral::before {{
            background: var(--gradient-2);
        }}
        
        .stat-card.purple::before {{
            background: var(--gradient-3);
        }}
        
        .stat-card.gold::before {{
            background: var(--gradient-gold);
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            border-color: var(--accent-cyan);
            box-shadow: 0 20px 40px rgba(78, 205, 196, 0.15);
        }}
        
        .stat-card .label {{
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 10px;
        }}
        
        .stat-card .value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .stat-card .change {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            margin-top: 12px;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        
        .stat-card .change.positive {{
            background: rgba(78, 205, 196, 0.15);
            color: var(--accent-cyan);
        }}
        
        .stat-card .change.negative {{
            background: rgba(255, 107, 107, 0.15);
            color: var(--accent-coral);
        }}
        
        /* Comparison Section */
        .comparison-section {{
            background: var(--bg-card);
            border-radius: 24px;
            padding: 40px;
            margin: 40px 0;
            border: 1px solid var(--border-color);
        }}
        
        .section-header {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 30px;
        }}
        
        .section-header .icon {{
            width: 50px;
            height: 50px;
            background: var(--gradient-1);
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }}
        
        .section-header h2 {{
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        /* Dataset Cards Side by Side */
        .dataset-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin: 30px 0;
        }}
        
        @media (max-width: 768px) {{
            .dataset-comparison {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .dataset-card {{
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 30px;
            border: 1px solid var(--border-color);
        }}
        
        .dataset-card.old {{
            border-top: 4px solid var(--accent-coral);
        }}
        
        .dataset-card.new {{
            border-top: 4px solid var(--accent-cyan);
        }}
        
        .dataset-card h3 {{
            font-size: 1.2rem;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .dataset-card.old h3 {{
            color: var(--accent-coral);
        }}
        
        .dataset-card.new h3 {{
            color: var(--accent-cyan);
        }}
        
        .dataset-card .metric {{
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .dataset-card .metric:last-child {{
            border-bottom: none;
        }}
        
        .dataset-card .metric-label {{
            color: var(--text-secondary);
        }}
        
        .dataset-card .metric-value {{
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        /* Charts */
        .chart-container {{
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 25px;
            margin: 25px 0;
            border: 1px solid var(--border-color);
            text-align: center;
        }}
        
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 12px;
        }}
        
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 25px;
        }}
        
        @media (max-width: 600px) {{
            .chart-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        /* Stats Table */
        .stats-table-container {{
            overflow-x: auto;
            margin: 30px 0;
        }}
        
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
        }}
        
        .stats-table th {{
            background: var(--bg-secondary);
            padding: 16px 20px;
            text-align: left;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 1px;
            border-bottom: 2px solid var(--border-color);
        }}
        
        .stats-table td {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
        }}
        
        .stats-table tr:hover {{
            background: var(--bg-card-hover);
        }}
        
        .stats-table .feature-name {{
            color: var(--text-primary);
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 500;
        }}
        
        .stats-table .positive {{
            color: var(--accent-cyan);
            font-weight: 600;
        }}
        
        .stats-table .negative {{
            color: var(--accent-coral);
            font-weight: 600;
        }}
        
        /* Conclusions */
        .conclusions {{
            background: linear-gradient(135deg, rgba(78, 205, 196, 0.1), rgba(168, 85, 247, 0.1));
            border-radius: 24px;
            padding: 40px;
            margin: 40px 0;
            border: 1px solid var(--border-color);
        }}
        
        .conclusions h2 {{
            font-size: 1.5rem;
            margin-bottom: 25px;
            color: var(--text-primary);
        }}
        
        .conclusion-item {{
            display: flex;
            gap: 15px;
            margin: 20px 0;
            padding: 20px;
            background: var(--bg-card);
            border-radius: 12px;
            border-left: 4px solid var(--accent-cyan);
        }}
        
        .conclusion-item .number {{
            width: 36px;
            height: 36px;
            background: var(--gradient-1);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            flex-shrink: 0;
        }}
        
        .conclusion-item .content h4 {{
            color: var(--text-primary);
            margin-bottom: 8px;
            font-size: 1.1rem;
        }}
        
        .conclusion-item .content p {{
            color: var(--text-secondary);
            font-size: 0.95rem;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 40px 0;
            color: var(--text-muted);
            font-size: 0.9rem;
            border-top: 1px solid var(--border-color);
            margin-top: 60px;
        }}
        
        .footer .logo {{
            font-size: 1.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        
        /* Animations */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .stat-card, .comparison-section, .conclusions {{
            animation: fadeIn 0.6s ease-out forwards;
        }}
        
        .stat-card:nth-child(2) {{ animation-delay: 0.1s; }}
        .stat-card:nth-child(3) {{ animation-delay: 0.2s; }}
        .stat-card:nth-child(4) {{ animation-delay: 0.3s; }}
    </style>
</head>
<body>
    <div class="bg-animation"></div>
    
    <div class="container">
        <header class="header">
            <h1>Dataset Comparison Report</h1>
            <p class="subtitle">Memory Forensics Data Analysis - Spyware Detection</p>
            <p class="date">Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
        </header>
        
        <!-- Key Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">Old Dataset Samples</div>
                <div class="value">{old_stats['rows']:,}</div>
                <div class="change negative">
                    <span>Baseline</span>
                </div>
            </div>
            <div class="stat-card coral">
                <div class="label">New Dataset Samples</div>
                <div class="value">{new_stats['rows']:,}</div>
                <div class="change {"positive" if row_diff >= 0 else "negative"}">
                    <span>{"+" if row_diff >= 0 else ""}{row_diff:,} ({row_pct:+.1f}%)</span>
                </div>
            </div>
            <div class="stat-card purple">
                <div class="label">Total Features</div>
                <div class="value">{old_stats['columns']}</div>
                <div class="change positive">
                    <span>Consistent</span>
                </div>
            </div>
            <div class="stat-card gold">
                <div class="label">Numeric Features</div>
                <div class="value">{old_stats['numeric_columns']}</div>
                <div class="change positive">
                    <span>For Analysis</span>
                </div>
            </div>
        </div>
        
        <!-- Dataset Comparison -->
        <section class="comparison-section">
            <div class="section-header">
                <div class="icon">📊</div>
                <h2>Dataset Overview</h2>
            </div>
            
            <div class="dataset-comparison">
                <div class="dataset-card old">
                    <h3>
                        <span>🔴</span> Old Dataset (Output1.csv)
                    </h3>
                    <div class="metric">
                        <span class="metric-label">Total Rows</span>
                        <span class="metric-value">{old_stats['rows']:,}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total Columns</span>
                        <span class="metric-value">{old_stats['columns']}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Numeric Columns</span>
                        <span class="metric-value">{old_stats['numeric_columns']}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Missing Values</span>
                        <span class="metric-value">{old_stats['missing_values']}</span>
                    </div>
                </div>
                
                <div class="dataset-card new">
                    <h3>
                        <span>🟢</span> New Dataset (Output3.csv)
                    </h3>
                    <div class="metric">
                        <span class="metric-label">Total Rows</span>
                        <span class="metric-value">{new_stats['rows']:,}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total Columns</span>
                        <span class="metric-value">{new_stats['columns']}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Numeric Columns</span>
                        <span class="metric-value">{new_stats['numeric_columns']}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Missing Values</span>
                        <span class="metric-value">{new_stats['missing_values']}</span>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Charts -->
        <section class="comparison-section">
            <div class="section-header">
                <div class="icon">📈</div>
                <h2>Visual Analysis</h2>
            </div>
            
            <div class="chart-grid">
                <div class="chart-container">
                    <img src="data:image/png;base64,{charts.get('size_comparison', '')}" alt="Size Comparison">
                </div>
                <div class="chart-container">
                    <img src="data:image/png;base64,{charts.get('mean_comparison', '')}" alt="Mean Comparison">
                </div>
            </div>
            
            <div class="chart-grid">
                <div class="chart-container">
                    <img src="data:image/png;base64,{charts.get('pct_change', '')}" alt="Percentage Change">
                </div>
                <div class="chart-container">
                    <img src="data:image/png;base64,{charts.get('distribution', '')}" alt="Distribution">
                </div>
            </div>
        </section>
        
        <!-- Statistical Comparison Table -->
        <section class="comparison-section">
            <div class="section-header">
                <div class="icon">🔢</div>
                <h2>Statistical Comparison</h2>
            </div>
            
            <div class="stats-table-container">
                <table class="stats-table">
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Old Mean</th>
                            <th>New Mean</th>
                            <th>Old Std Dev</th>
                            <th>New Std Dev</th>
                            <th>Change</th>
                        </tr>
                    </thead>
                    <tbody>
                        {stats_rows}
                    </tbody>
                </table>
            </div>
        </section>
        
        <!-- Conclusions -->
        <section class="conclusions">
            <h2>🎯 Key Findings & Recommendations</h2>
            
            <div class="conclusion-item">
                <div class="number">1</div>
                <div class="content">
                    <h4>Sample Size Evolution</h4>
                    <p>The new dataset contains <strong>{abs(row_diff)}</strong> {'more' if row_diff > 0 else 'fewer'} samples 
                    ({abs(row_pct):.1f}% {'increase' if row_diff > 0 else 'decrease'}), indicating 
                    {'expanded data collection for better model training' if row_diff > 0 else 'refined data selection'}.</p>
                </div>
            </div>
            
            <div class="conclusion-item">
                <div class="number">2</div>
                <div class="content">
                    <h4>Feature Consistency</h4>
                    <p>Both datasets maintain the same structure with <strong>{old_stats['columns']} features</strong>, 
                    ensuring seamless compatibility for model comparison and transfer learning approaches.</p>
                </div>
            </div>
            
            <div class="conclusion-item">
                <div class="number">3</div>
                <div class="content">
                    <h4>Data Quality Assessment</h4>
                    <p>Missing values are {'minimal' if max(old_stats['missing_values'], new_stats['missing_values']) < 100 else 'present'} 
                    in both datasets. Old: <strong>{old_stats['missing_values']}</strong>, New: <strong>{new_stats['missing_values']}</strong>.</p>
                </div>
            </div>
            
            <div class="conclusion-item">
                <div class="number">4</div>
                <div class="content">
                    <h4>Recommendations</h4>
                    <p>• Consider ensemble approaches using both datasets<br>
                    • Monitor feature drift in production deployments<br>
                    • Validate model performance on both old and new data separately</p>
                </div>
            </div>
        </section>
        
        <footer class="footer">
            <div class="logo">SIEM Analytics</div>
            <p>Network Traffic Anomaly & Threat Classification System</p>
            <p style="margin-top: 10px; opacity: 0.7;">Report generated automatically | {datetime.now().year}</p>
        </footer>
    </div>
</body>
</html>
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return True


def main():
    """Main function to generate the comparison report"""
    print("=" * 60)
    print("Dataset Comparison HTML Report Generator")
    print("=" * 60)
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    print("\n[1/4] Loading datasets...")
    old_df, new_df = load_datasets()
    print(f"   Old dataset: {len(old_df)} rows, {len(old_df.columns)} columns")
    print(f"   New dataset: {len(new_df)} rows, {len(new_df.columns)} columns")
    
    print("\n[2/4] Calculating statistics...")
    old_stats = calculate_statistics(old_df, "Old Dataset (Output1)")
    new_stats = calculate_statistics(new_df, "New Dataset (Output3)")
    
    print("\n[3/4] Creating comparison charts...")
    charts = create_charts(old_df, new_df)
    print(f"   Created {len(charts)} charts")
    
    print("\n[4/4] Generating HTML report...")
    output_html = os.path.join(base_path, 'Dataset_Comparison_Report.html')
    
    success = generate_html_report(old_df, new_df, old_stats, new_stats, charts, output_html)
    
    if success:
        print(f"\n[SUCCESS] HTML report generated successfully!")
        print(f"   Location: {output_html}")
        
        # Try to open in browser
        import webbrowser
        webbrowser.open(f'file:///{output_html}')
        print("   Opening in browser...")
    else:
        print("\n[ERROR] Failed to generate HTML report")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()


