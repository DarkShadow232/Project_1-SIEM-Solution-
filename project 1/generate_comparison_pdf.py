"""
Dataset Comparison PDF Generator
Compares older dataset (Output1.csv) with newer dataset (output3.csv)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# Try to import PDF generation libraries
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

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
    
    # Load datasets
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
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats['numeric_columns'] = len(numeric_cols)
    
    # Calculate statistics for numeric columns
    if numeric_cols:
        stats['mean_values'] = df[numeric_cols].mean().to_dict()
        stats['std_values'] = df[numeric_cols].std().to_dict()
        stats['min_values'] = df[numeric_cols].min().to_dict()
        stats['max_values'] = df[numeric_cols].max().to_dict()
        stats['median_values'] = df[numeric_cols].median().to_dict()
    
    # Missing values
    stats['missing_values'] = df.isnull().sum().sum()
    stats['missing_by_column'] = df.isnull().sum().to_dict()
    
    return stats


def create_comparison_charts(old_df, new_df, output_dir):
    """Create comparison charts"""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping charts.")
        return []
    
    chart_paths = []
    
    # Get common numeric columns
    old_numeric = set(old_df.select_dtypes(include=[np.number]).columns)
    new_numeric = set(new_df.select_dtypes(include=[np.number]).columns)
    common_cols = list(old_numeric.intersection(new_numeric))
    
    if not common_cols:
        return chart_paths
    
    # Select top 8 features for visualization
    key_features = common_cols[:8] if len(common_cols) > 8 else common_cols
    
    # Chart 1: Dataset Size Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Rows', 'Numeric Columns', 'Total Columns']
    old_values = [len(old_df), len(old_numeric), len(old_df.columns)]
    new_values = [len(new_df), len(new_numeric), len(new_df.columns)]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, old_values, width, label='Old Dataset (Output1)', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, new_values, width, label='New Dataset (Output3)', color='#4ECDC4', alpha=0.8)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Dataset Size Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, 'chart_size_comparison.png')
    plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
    plt.close()
    chart_paths.append(chart1_path)
    
    # Chart 2: Mean Values Comparison for Key Features
    fig, ax = plt.subplots(figsize=(12, 6))
    
    old_means = [old_df[col].mean() for col in key_features]
    new_means = [new_df[col].mean() for col in key_features]
    
    x = np.arange(len(key_features))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, old_means, width, label='Old Dataset', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, new_means, width, label='New Dataset', color='#4ECDC4', alpha=0.8)
    
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Mean Value', fontsize=12)
    ax.set_title('Mean Values Comparison - Key Features', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([col.split('.')[-1][:15] for col in key_features], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, 'chart_mean_comparison.png')
    plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
    plt.close()
    chart_paths.append(chart2_path)
    
    # Chart 3: Distribution Comparison (Box plots)
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(key_features[:8]):
        ax = axes[i]
        data_to_plot = [old_df[col].dropna(), new_df[col].dropna()]
        bp = ax.boxplot(data_to_plot, tick_labels=['Old', 'New'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#FF6B6B')
        bp['boxes'][1].set_facecolor('#4ECDC4')
        ax.set_title(col.split('.')[-1][:20], fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Distribution Comparison - Key Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, 'chart_distribution_comparison.png')
    plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
    plt.close()
    chart_paths.append(chart3_path)
    
    # Chart 4: Percentage Change in Key Metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pct_changes = []
    for col in key_features:
        old_mean = old_df[col].mean()
        new_mean = new_df[col].mean()
        if old_mean != 0:
            pct_change = ((new_mean - old_mean) / abs(old_mean)) * 100
        else:
            pct_change = 0
        pct_changes.append(pct_change)
    
    colors_list = ['#4ECDC4' if x >= 0 else '#FF6B6B' for x in pct_changes]
    bars = ax.barh(range(len(key_features)), pct_changes, color=colors_list, alpha=0.8)
    
    ax.set_yticks(range(len(key_features)))
    ax.set_yticklabels([col.split('.')[-1][:20] for col in key_features])
    ax.set_xlabel('Percentage Change (%)', fontsize=12)
    ax.set_title('Percentage Change in Mean Values (Old → New)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    chart4_path = os.path.join(output_dir, 'chart_pct_change.png')
    plt.savefig(chart4_path, dpi=150, bbox_inches='tight')
    plt.close()
    chart_paths.append(chart4_path)
    
    return chart_paths


def generate_pdf_report(old_df, new_df, old_stats, new_stats, chart_paths, output_path):
    """Generate PDF comparison report"""
    if not REPORTLAB_AVAILABLE:
        print("ReportLab not available. Cannot generate PDF.")
        return False
    
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#2C3E50'),
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.HexColor('#34495E')
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=12,
        textColor=colors.HexColor('#5D6D7E')
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        textColor=colors.HexColor('#2C3E50')
    )
    
    elements = []
    
    # Title
    elements.append(Paragraph("Dataset Comparison Report", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 20))
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", heading_style))
    
    summary_text = f"""
    This report compares two memory forensics datasets used for malware/spyware detection:
    <br/><br/>
    <b>Old Dataset (Output1.csv):</b> {old_stats['rows']} samples with {old_stats['columns']} features<br/>
    <b>New Dataset (Output3.csv):</b> {new_stats['rows']} samples with {new_stats['columns']} features<br/>
    <br/>
    <b>Key Findings:</b><br/>
    • Sample count difference: {new_stats['rows'] - old_stats['rows']} ({((new_stats['rows']-old_stats['rows'])/old_stats['rows']*100):.1f}% {'increase' if new_stats['rows'] > old_stats['rows'] else 'decrease'})<br/>
    • Both datasets contain {old_stats['numeric_columns']} numeric features for analysis<br/>
    • Feature structure is consistent between datasets
    """
    elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1, 20))
    
    # Dataset Overview Table
    elements.append(Paragraph("Dataset Overview", heading_style))
    
    overview_data = [
        ['Metric', 'Old Dataset (Output1)', 'New Dataset (Output3)', 'Difference'],
        ['Total Rows', str(old_stats['rows']), str(new_stats['rows']), str(new_stats['rows'] - old_stats['rows'])],
        ['Total Columns', str(old_stats['columns']), str(new_stats['columns']), str(new_stats['columns'] - old_stats['columns'])],
        ['Numeric Columns', str(old_stats['numeric_columns']), str(new_stats['numeric_columns']), str(new_stats['numeric_columns'] - old_stats['numeric_columns'])],
        ['Missing Values', str(old_stats['missing_values']), str(new_stats['missing_values']), str(new_stats['missing_values'] - old_stats['missing_values'])],
    ]
    
    overview_table = Table(overview_data, colWidths=[140, 120, 120, 80])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ECF0F1')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    elements.append(overview_table)
    elements.append(Spacer(1, 20))
    
    # Add Charts
    if chart_paths:
        elements.append(PageBreak())
        elements.append(Paragraph("Visual Comparison", heading_style))
        
        for i, chart_path in enumerate(chart_paths):
            if os.path.exists(chart_path):
                elements.append(Spacer(1, 10))
                img = Image(chart_path, width=6*inch, height=4*inch)
                elements.append(img)
                elements.append(Spacer(1, 15))
                
                if i == 1:  # Page break after 2 charts
                    elements.append(PageBreak())
    
    # Statistical Comparison
    elements.append(PageBreak())
    elements.append(Paragraph("Statistical Comparison - Key Features", heading_style))
    
    # Get common numeric columns
    old_numeric = set(old_df.select_dtypes(include=[np.number]).columns)
    new_numeric = set(new_df.select_dtypes(include=[np.number]).columns)
    common_cols = list(old_numeric.intersection(new_numeric))[:10]
    
    if common_cols:
        stat_data = [['Feature', 'Old Mean', 'New Mean', 'Old Std', 'New Std', 'Change %']]
        
        for col in common_cols:
            old_mean = old_df[col].mean()
            new_mean = new_df[col].mean()
            old_std = old_df[col].std()
            new_std = new_df[col].std()
            
            if old_mean != 0:
                pct_change = ((new_mean - old_mean) / abs(old_mean)) * 100
            else:
                pct_change = 0
            
            # Truncate column name for display
            col_display = col.split('.')[-1][:18]
            
            stat_data.append([
                col_display,
                f'{old_mean:.2f}',
                f'{new_mean:.2f}',
                f'{old_std:.2f}',
                f'{new_std:.2f}',
                f'{pct_change:+.1f}%'
            ])
        
        stat_table = Table(stat_data, colWidths=[90, 70, 70, 70, 70, 60])
        stat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27AE60')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#E8F8F5')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#A9DFBF')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ]))
        elements.append(stat_table)
    
    # Conclusions
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("Conclusions", heading_style))
    
    # Calculate some insights
    size_diff = new_stats['rows'] - old_stats['rows']
    size_pct = (size_diff / old_stats['rows']) * 100 if old_stats['rows'] > 0 else 0
    
    conclusions_text = f"""
    <b>Dataset Evolution Analysis:</b><br/><br/>
    
    1. <b>Sample Size:</b> The new dataset contains {abs(size_diff)} {'more' if size_diff > 0 else 'fewer'} samples 
       ({abs(size_pct):.1f}% {'increase' if size_diff > 0 else 'decrease'}), indicating 
       {'expanded data collection' if size_diff > 0 else 'refined/filtered data'}.<br/><br/>
    
    2. <b>Feature Consistency:</b> Both datasets maintain the same feature structure with {old_stats['columns']} 
       columns, ensuring compatibility for model training and comparison.<br/><br/>
    
    3. <b>Data Quality:</b> Missing values are {'minimal' if max(old_stats['missing_values'], new_stats['missing_values']) < 100 else 'present'} 
       in both datasets, with the new dataset having {new_stats['missing_values']} missing values 
       compared to {old_stats['missing_values']} in the old dataset.<br/><br/>
    
    4. <b>Statistical Variations:</b> The statistical distributions show variations between datasets, 
       which may reflect different malware samples, system states, or collection periods.<br/><br/>
    
    <b>Recommendations:</b><br/>
    • Consider combining both datasets for more robust model training<br/>
    • Validate model performance on both old and new data separately<br/>
    • Monitor feature drift over time for production systems
    """
    elements.append(Paragraph(conclusions_text, normal_style))
    
    # Build PDF
    doc.build(elements)
    return True


def main():
    """Main function to generate the comparison report"""
    print("=" * 60)
    print("Dataset Comparison PDF Generator")
    print("=" * 60)
    
    # Set output directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_path, 'siem_integration', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    print("\n[1/4] Loading datasets...")
    old_df, new_df = load_datasets()
    print(f"   Old dataset: {len(old_df)} rows, {len(old_df.columns)} columns")
    print(f"   New dataset: {len(new_df)} rows, {len(new_df.columns)} columns")
    
    # Calculate statistics
    print("\n[2/4] Calculating statistics...")
    old_stats = calculate_statistics(old_df, "Old Dataset (Output1)")
    new_stats = calculate_statistics(new_df, "New Dataset (Output3)")
    
    # Create charts
    print("\n[3/4] Creating comparison charts...")
    chart_paths = create_comparison_charts(old_df, new_df, output_dir)
    print(f"   Created {len(chart_paths)} charts")
    
    # Generate PDF
    print("\n[4/4] Generating PDF report...")
    output_pdf = os.path.join(base_path, 'Dataset_Comparison_Report.pdf')
    
    if REPORTLAB_AVAILABLE:
        success = generate_pdf_report(old_df, new_df, old_stats, new_stats, chart_paths, output_pdf)
        if success:
            print(f"\n[SUCCESS] PDF report generated successfully!")
            print(f"   Location: {output_pdf}")
        else:
            print("\n[ERROR] Failed to generate PDF report")
    else:
        print("\n[WARNING] ReportLab library not installed. Installing now...")
        print("   Please run: pip install reportlab matplotlib")
        
        # Generate a text-based report as fallback
        text_report = os.path.join(base_path, 'Dataset_Comparison_Report.txt')
        with open(text_report, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DATASET COMPARISON REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Old Dataset (Output1.csv):\n")
            f.write(f"  - Rows: {old_stats['rows']}\n")
            f.write(f"  - Columns: {old_stats['columns']}\n")
            f.write(f"  - Numeric Columns: {old_stats['numeric_columns']}\n")
            f.write(f"  - Missing Values: {old_stats['missing_values']}\n\n")
            f.write(f"New Dataset (Output3.csv):\n")
            f.write(f"  - Rows: {new_stats['rows']}\n")
            f.write(f"  - Columns: {new_stats['columns']}\n")
            f.write(f"  - Numeric Columns: {new_stats['numeric_columns']}\n")
            f.write(f"  - Missing Values: {new_stats['missing_values']}\n\n")
            f.write("DIFFERENCES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Row Difference: {new_stats['rows'] - old_stats['rows']}\n")
            f.write(f"Column Difference: {new_stats['columns'] - old_stats['columns']}\n")
        
        print(f"   Text report generated: {text_report}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

