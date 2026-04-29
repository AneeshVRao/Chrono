"""
Automated Daily Report Generator

Generates a sleek HTML tear-sheet summarizing the current status of the trading system,
including current positions, latest model confidence scores, and system health.

Usage:
    python scripts/generate_daily_report.py
"""

import os
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.utils.logger import get_logger
from src.utils.config_loader import Config

logger = get_logger(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chrono Quant | Daily Tear Sheet</title>
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background-color: #1e293b;
            border-radius: 12px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
            font-weight: 700;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .section-title {{
            font-size: 20px;
            font-weight: 600;
            margin-top: 0;
            margin-bottom: 20px;
            border-bottom: 1px solid #334155;
            padding-bottom: 10px;
            color: #94a3b8;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background-color: #0f172a;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #334155;
        }}
        .card-value {{
            font-size: 24px;
            font-weight: 700;
            margin-top: 10px;
            color: #38bdf8;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #334155;
        }}
        th {{
            background-color: #0f172a;
            color: #94a3b8;
            font-weight: 600;
        }}
        .positive {{ color: #10b981; }}
        .negative {{ color: #ef4444; }}
        .neutral {{ color: #94a3b8; }}
        .footer {{
            text-align: center;
            padding: 20px;
            font-size: 14px;
            color: #64748b;
            border-top: 1px solid #334155;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Chrono Quant AI</h1>
            <p>Daily Execution Tear Sheet &bull; {date}</p>
        </div>
        
        <div class="content">
            <h2 class="section-title">System Status</h2>
            <div class="grid">
                <div class="card">
                    <div>Data Versioning</div>
                    <div class="card-value">{data_version}</div>
                </div>
                <div class="card">
                    <div>Analyzed Universe</div>
                    <div class="card-value">{universe_size} Assets</div>
                </div>
            </div>

            <h2 class="section-title">Latest Model Signals</h2>
            <table>
                <thead>
                    <tr>
                        <th>Asset</th>
                        <th>ML Confidence</th>
                        <th>Meta-Model Status</th>
                        <th>Recommended Target</th>
                    </tr>
                </thead>
                <tbody>
                    {signal_rows}
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            Generated automatically by Chrono Quant Pipeline
        </div>
    </div>
</body>
</html>
"""


def generate_report():
    logger.info("Generating Daily HTML Report...")
    cfg = Config()
    
    date_str = datetime.now().strftime("%B %d, %Y - %H:%M UTC")
    
    # 1. Get Data Version
    version_file = cfg.features_dir / "feature_metadata.json"
    data_version = "Unknown"
    universe_size = len(cfg.tickers)
    
    if version_file.exists():
        import json
        with open(version_file, "r") as f:
            meta = json.load(f)
            data_version = meta.get("version", "Unknown")[:10]
            
    # 2. Get latest signals from features
    signal_rows = ""
    for ticker in cfg.tickers:
        feature_path = cfg.features_dir / f"{ticker}_features.parquet"
        if not feature_path.exists():
            continue
            
        try:
            df = pd.read_parquet(feature_path, engine="pyarrow")
            if df.empty:
                continue
                
            last_row = df.iloc[-1]
            proba = last_row.get("proba_Ensemble", "N/A")
            meta = last_row.get("meta_pred", 1.0)
            
            if isinstance(proba, float) and not pd.isna(proba):
                edge = 2.0 * proba - 1.0
                target = max(0.0, 0.5 * edge) * meta
                proba_str = f"{proba:.2%}"
            else:
                target = 0.0
                proba_str = "N/A"
                
            meta_status = "Approved" if meta > 0 else "Blocked"
            meta_color = "positive" if meta > 0 else "negative"
            
            target_str = f"{target:.2%}"
            target_color = "positive" if target > 0 else "neutral"
            
            signal_rows += f'''
            <tr>
                <td><strong>{ticker}</strong></td>
                <td>{proba_str}</td>
                <td class="{meta_color}">{meta_status}</td>
                <td class="{target_color}">{target_str}</td>
            </tr>
            '''
        except Exception as e:
            logger.warning(f"Error reading {ticker} for report: {e}")

    html_content = HTML_TEMPLATE.format(
        date=date_str,
        data_version=data_version,
        universe_size=universe_size,
        signal_rows=signal_rows
    )
    
    report_dir = Path("logs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / f"daily_tearsheet_{datetime.now().strftime('%Y%m%d')}.html"
    with open(report_path, "w") as f:
        f.write(html_content)
        
    logger.info(f"Report generated successfully: {report_path}")


if __name__ == "__main__":
    generate_report()
