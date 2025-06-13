import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import json
import math
from typing import Dict, List, Tuple, Optional, Any

# Import the existing modules
from .adcp_parser import parse_file, save_tables_html
from .adcp_grids import build_metric_grids
from .adcp_stats import compare_bins, validity_report, histogram_validity, _slice_window

def convert_to_json_safe(obj: Any) -> Any:
    """Convert any object to JSON-safe format, handling all edge cases"""
    
    # Handle None
    if obj is None:
        return None
    
    # Handle numpy/pandas NaN, Infinity
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj):
            return None  # Convert NaN to null
        elif math.isinf(obj):
            return None  # Convert Infinity to null
        else:
            return float(obj)
    
    # Handle numpy integers
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    
    # Handle numpy floats
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return [convert_to_json_safe(item) for item in obj.tolist()]
    
    # Handle pandas Series
    if isinstance(obj, pd.Series):
        return convert_to_json_safe(obj.to_dict())
    
    # Handle pandas DataFrames
    if isinstance(obj, pd.DataFrame):
        return convert_to_json_safe(obj.to_dict())
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {str(key): convert_to_json_safe(value) for key, value in obj.items()}
    
    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_safe(item) for item in obj]
    
    # Handle strings - ensure they're proper strings
    if isinstance(obj, str):
        return obj
    
    # Handle booleans
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    
    # Handle regular Python numbers
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    
    # For anything else, try to convert to string
    try:
        return str(obj)
    except:
        return None

def validate_json_serializable(obj: Any) -> bool:
    """Test if an object can be JSON serialized"""
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError, OverflowError):
        return False

class ADCPAnalyzer:
    def __init__(self):
        self.ca = None
        self.cs = None
        self.speed_mat = None
        self.dir_mat = None
    
    def analyze_file(self, filepath: str) -> Dict:
        """Main analysis function that processes the ADCP file and returns all results"""
        try:
            # Parse the file
            print("🔍 Parsing ADCP file...")
            self.ca, self.cs = parse_file(filepath)
            print(f"✅ Parsed - Aquadopp: {len(self.ca)} records, Signature: {len(self.cs)} records")
            
            # Build metric grids
            print("📊 Building metric grids...")
            self.speed_mat, self.dir_mat = build_metric_grids(self.ca, self.cs, max_bin=29)
            print(f"✅ Built grids - Speed matrix: {self.speed_mat.shape}")
            
            # Generate all analysis results
            print("📈 Generating analysis results...")
            results = {
                'basic_info': self._get_basic_info(),
                'bin_comparison': self._compare_bins_analysis(),
                'validity_report': self._validity_analysis(),
                'hourly_analysis': self._hourly_analysis(),
                'correlation_analysis': self._correlation_analysis(),
                'interactive_plots': self._generate_interactive_plots()
            }
            
            print("🔄 Converting to JSON-safe format...")
            # Convert to JSON-safe format
            safe_results = convert_to_json_safe(results)
            
            # Validate that the result is JSON serializable
            print("✅ Validating JSON serializability...")
            if not validate_json_serializable(safe_results):
                raise Exception("Results contain non-JSON-serializable data")
            
            print("🎉 Analysis completed successfully!")
            return safe_results
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Error in analysis: {str(e)}")
    
    def _get_basic_info(self) -> Dict:
        """Get basic information about the dataset"""
        print("   📋 Getting basic info...")
        info = {
            'aquadopp_records': len(self.ca),
            'signature_records': len(self.cs),
            'time_range': {
                'start': str(self.speed_mat.columns.min()),
                'end': str(self.speed_mat.columns.max())
            },
            'total_bins': len(self.speed_mat.index),
            'data_summary': self.speed_mat.describe().to_html(classes='table table-striped')
        }
        return info
    
    def _compare_bins_analysis(self) -> Dict:
        """Compare bins analysis from the notebook"""
        print("   📊 Comparing bins...")
        bins = ["BIN0", "BIN1"]
        
        # Generate the comparison plot
        fig = plt.figure(figsize=(12, 8))
        
        # Get data for the specified bins
        window = self.speed_mat.reindex(index=bins)
        
        # Count valid points
        valid_counts = {}
        for b in bins:
            finite = window.loc[b].notna().sum()
            valid_counts[b] = int(finite)  # Ensure regular int
        
        # Create the plot
        colors = ["#e74c3c", "#3498db", "#2ecc71"]
        for i, bin_name in enumerate(bins):
            data = window.loc[bin_name].dropna()
            plt.plot(data.index, data.values, 
                    color=colors[i], label=f'{bin_name} ({valid_counts[bin_name]} points)',
                    linewidth=1.5, markersize=2, marker='o')
        
        plt.title('Speed Comparison Between Bins')
        plt.xlabel('Timestamp')
        plt.ylabel('Speed (m/s)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return {
            'plot': img_str,
            'valid_counts': valid_counts,
            'comparison_bins': bins
        }
    
    def _validity_analysis(self) -> Dict:
        """Generate validity report and histogram"""
        print("   ✅ Analyzing validity...")
        # Validity report
        report_df = validity_report(self.speed_mat)
        
        # Histogram of validity
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate validity percentages for each bin
        validity_pcts = []
        
        for bin_name in self.speed_mat.index:
            total_points = len(self.speed_mat.columns)
            valid_points = self.speed_mat.loc[bin_name].notna().sum()
            validity_pct = float((valid_points / total_points) * 100)
            
            # Handle any potential NaN
            if math.isnan(validity_pct):
                validity_pct = 0.0
            
            validity_pcts.append(validity_pct)
        
        # Create histogram
        bins_hist = np.arange(0, 101, 5)  # 0-100% in 5% bins
        ax.hist(validity_pcts, bins=bins_hist, edgecolor='black', alpha=0.7, color='skyblue')
        ax.set_xlabel('Validity Percentage (%)')
        ax.set_ylabel('Number of Bins')
        ax.set_title('Distribution of Data Validity Across Bins')
        ax.grid(True, alpha=0.3)
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        hist_img = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        # Calculate stats safely
        avg_validity = float(np.mean(validity_pcts))
        min_validity = float(np.min(validity_pcts))
        max_validity = float(np.max(validity_pcts))
        
        # Handle any NaN values
        if math.isnan(avg_validity): avg_validity = 0.0
        if math.isnan(min_validity): min_validity = 0.0
        if math.isnan(max_validity): max_validity = 0.0
        
        return {
            'report_table': report_df.to_html(classes='table table-striped'),
            'histogram': hist_img,
            'avg_validity': avg_validity,
            'min_validity': min_validity,
            'max_validity': max_validity
        }
    
    def _hourly_analysis(self) -> Dict:
        """Analyze invalid data distribution by hour"""
        print("   🕐 Analyzing hourly patterns...")
        # Count invalid bins at each timestamp
        invalid_per_ts = self.speed_mat.isna().sum(axis=0)
        
        # Aggregate by hour of day
        hourly_sum = invalid_per_ts.groupby(invalid_per_ts.index.hour).sum()
        
        # Create separate analysis for Signature and Aquadopp
        signature_bins = [f"BIN{i}" for i in range(1, 30)]
        aquadopp_bins = ["BIN0"]
        
        hourly_data = {}
        
        for label, bins in [("Signature (BIN1-29)", signature_bins), ("Aquadopp (BIN0)", aquadopp_bins)]:
            # Filter bins that actually exist
            existing_bins = [b for b in bins if b in self.speed_mat.index]
            if not existing_bins:
                hourly_data[label] = {}
                continue
                
            win = self.speed_mat.loc[existing_bins]
            invalid_per_ts_subset = win.isna().sum(axis=0)
            hourly_sum_subset = invalid_per_ts_subset.groupby(invalid_per_ts_subset.index.hour).sum()
            # Convert to safe dict
            hourly_data[label] = {int(k): int(v) for k, v in hourly_sum_subset.to_dict().items()}
        
        # Create side-by-side plot
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5), sharey=True)
        colors = {"Signature (BIN1-29)": "#c0392b", "Aquadopp (BIN0)": "#2980b9"}
        
        for ax, label in zip(axes, hourly_data.keys()):
            if hourly_data[label]:  # Only plot if data exists
                hours = list(hourly_data[label].keys())
                values = list(hourly_data[label].values())
                ax.bar(hours, values, color=colors[label])
            ax.set_title(label)
            ax.set_xlabel('Hour of Day')
            if ax == axes[0]:
                ax.set_ylabel('Total Invalid Bins')
            ax.grid(True, axis='y', alpha=0.3)
        
        fig.suptitle('Invalid Data Distribution by Hour of Day', fontsize=14)
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return {
            'plot': img_str,
            'hourly_data': hourly_data,
            'total_hourly': {int(k): int(v) for k, v in hourly_sum.to_dict().items()}
        }
    
    def _correlation_analysis(self) -> Dict:
        """Analyze correlation of invalid data between bins"""
        print("   🔗 Analyzing correlations...")
        # Create boolean matrix for invalid data
        invalid = self.speed_mat.isna()
        bins = invalid.index
        
        # Calculate conditional probabilities
        cond_df = pd.DataFrame(index=bins, columns=bins, dtype=float)
        
        for i in bins:
            denom = invalid.loc[i].sum()
            if denom == 0:
                cond_df.loc[i] = np.nan
                continue
            for j in bins:
                joint = (invalid.loc[i] & invalid.loc[j]).sum()
                cond_df.loc[i, j] = joint / denom
        
        # Calculate Pearson correlation of invalid flags
        phi_df = invalid.T.corr(method="pearson")
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(phi_df, vmin=-1, vmax=1, cmap="coolwarm",
                   square=True, cbar_kws={"label": "φ / Pearson r"})
        plt.title("Correlation of Invalid Occurrences Across Bins")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        heatmap_img = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        # Get correlation stats safely
        triu_values = phi_df.values[np.triu_indices_from(phi_df.values, k=1)]
        # Remove any NaN values
        triu_values = triu_values[~np.isnan(triu_values)]
        
        if len(triu_values) > 0:
            mean_corr = float(np.mean(triu_values))
            max_corr = float(np.max(triu_values))
            min_corr = float(np.min(triu_values))
        else:
            mean_corr = max_corr = min_corr = 0.0
        
        return {
            'correlation_heatmap': heatmap_img,
            'conditional_prob_table': cond_df.to_html(classes='table table-striped'),
            'correlation_stats': {
                'mean_correlation': mean_corr,
                'max_correlation': max_corr,
                'min_correlation': min_corr
            }
        }
    
    def _generate_interactive_plots(self) -> Dict:
        """Generate interactive Plotly plots"""
        print("   🎨 Generating interactive plots...")
        
        # Interactive bin comparison
        available_bins = [b for b in ["BIN0", "BIN1", "BIN5"] if b in self.speed_mat.index]
        if not available_bins:
            available_bins = list(self.speed_mat.index[:3])  # Use first 3 available bins
        
        df = self.speed_mat.reindex(index=available_bins)
        
        tidy = (
            df.T.reset_index()
            .melt(id_vars="index", var_name="bin", value_name="speed_ms")
            .rename(columns={"index": "timestamp"})
        )
        
        # Remove any NaN values for plotly
        tidy = tidy.dropna()
        
        fig_interactive = px.line(
            tidy,
            x="timestamp",
            y="speed_ms", 
            color="bin",
            title="Interactive Speed Comparison",
            markers=True
        )
        fig_interactive.update_traces(connectgaps=False, marker_size=4)
        fig_interactive.update_yaxes(title="Speed (m/s)")
        fig_interactive.update_xaxes(title="Timestamp")
        
        # Invalid data visualization
        invalid = self.speed_mat.isna().astype(int)
        
        fig_invalid = px.imshow(
            invalid,
            aspect="auto",
            color_continuous_scale=[[0, "green"], [1, "red"]],
            origin="upper",
            labels=dict(x="Timestamp", y="Bin", color="Invalid"),
            title="Invalid Data Map (Red = Invalid, Green = Valid)"
        )
        fig_invalid.update_yaxes(autorange="reversed")
        fig_invalid.update_xaxes(tickangle=-45)
        fig_invalid.update_layout(coloraxis_showscale=False, height=700)
        
        return {
            'speed_comparison': fig_interactive.to_html(div_id="speed_comparison"),
            'invalid_data_map': fig_invalid.to_html(div_id="invalid_data_map")
        }
