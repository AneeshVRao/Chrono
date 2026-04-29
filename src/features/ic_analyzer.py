from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils.logger import get_logger

logger = get_logger(__name__)

class ICAnalyzer:
    """
    Information Coefficient (IC) Analyzer.
    Evaluates signal quality by correlating features at time t with return at time t+h.
    Ensures strictly no lookahead bias by shifting forward returns back to matching feature index.
    """

    def __init__(self, data: pd.DataFrame, horizons: List[int] = [1, 5, 10, 20]):
        """
        Args:
            data: DataFrame with features and price columns.
            horizons: List of forward periods to compute IC for.
        """
        self.df = data.copy()
        self.horizons = horizons
        self.ic_results: pd.DataFrame | None = None
        self.half_lives: Dict[str, float] = {}

    def _compute_forward_returns(self, price_col: str = 'close') -> pd.DataFrame:
        """Compute forward looking returns ensuring no lookahead leakage."""
        returns = pd.DataFrame(index=self.df.index)
        for h in self.horizons:
            # Shift (-h) brings future return to current time t
            # return_t+h = (price_t+h / price_t) - 1
            returns[f'fwd_ret_{h}d'] = self.df[price_col].pct_change(periods=h).shift(-h)
        return returns

    def compute_ic(self, feature_cols: List[str], price_col: str = 'close', method: str = 'spearman') -> pd.DataFrame:
        """
        Compute Information Coefficient (Spearman Rank Correlation) for multi-horizons.
        
        Args:
            feature_cols: Features to evaluate.
            price_col: Column used for computing returns.
            method: 'spearman' for rank IC (preferred, robust to outliers) or 'pearson'.
            
        Returns:
            DataFrame of IC values across horizons.
        """
        logger.info(f"Computing {method.capitalize()} IC for {len(feature_cols)} features...")
        fwd_returns = self._compute_forward_returns(price_col)
        
        ic_dict = {}
        for feature in feature_cols:
            if feature not in self.df.columns:
                continue
                
            ic_h = {}
            for h in self.horizons:
                ret_col = f'fwd_ret_{h}d'
                
                # Filter valid overlapping data points
                valid_mask = self.df[feature].notna() & fwd_returns[ret_col].notna()
                
                # Remove outliers for stable IC computation (keeps values inside 1-99 percentile)
                if valid_mask.sum() > 30:
                    v_feat = self.df.loc[valid_mask, feature]
                    v_ret = fwd_returns.loc[valid_mask, ret_col]
                    
                    # Compute IC
                    ic = v_feat.corr(v_ret, method=method)
                    ic_h[f'IC({h}d)'] = ic
                else:
                    ic_h[f'IC({h}d)'] = np.nan
                    
            ic_dict[feature] = ic_h
            
        self.ic_results = pd.DataFrame.from_dict(ic_dict, orient='index')
        return self.ic_results

    def filter_features(self, threshold: float = 0.01) -> Tuple[List[str], List[str]]:
        """
        Flag features with weak predictive power based on 1-day absolute IC.
        
        Args:
            threshold: Minimum required |IC| to pass.
            
        Returns:
            Tuple of (strong_features, weak_features).
        """
        if self.ic_results is None:
            raise ValueError("Must compute IC first via compute_ic().")
            
        if 'IC(1d)' not in self.ic_results.columns:
            return list(self.ic_results.index), []
            
        abs_ic = self.ic_results['IC(1d)'].abs()
        weak_features = self.ic_results[abs_ic < threshold].index.tolist()
        strong_features = self.ic_results[abs_ic >= threshold].index.tolist()
        
        logger.info(f"Feature Filtering (Threshold |IC| >= {threshold}):")
        logger.info(f"  - Strong signals: {len(strong_features)}")
        logger.info(f"  - Weak signals flagged: {len(weak_features)}")
        
        return strong_features, weak_features

    def compute_decay_half_life(self) -> Dict[str, float]:
        """
        Calculate the approximate half-life of each signal (when |IC| drops to 50% of 1d IC).
        """
        if self.ic_results is None:
            raise ValueError("Must compute IC first via compute_ic().")
            
        half_lives = {}
        for feature in self.ic_results.index:
            ic_vals = self.ic_results.loc[feature].abs()
            ic_1d = ic_vals.get('IC(1d)', np.nan)
            
            if pd.isna(ic_1d) or ic_1d == 0:
                half_lives[feature] = np.nan
                continue
                
            target = ic_1d / 2.0
            hl = np.nan
            
            # Find the crossing point
            for i in range(1, len(self.horizons)):
                h_prev, h_curr = self.horizons[i-1], self.horizons[i]
                v_prev, v_curr = ic_vals.iloc[i-1], ic_vals.iloc[i]
                
                if v_curr <= target <= v_prev:
                    # Linear interpolation for precise half-life estimation
                    slope = (v_curr - v_prev) / (h_curr - h_prev) if h_curr != h_prev else 0
                    if slope != 0:
                        hl = h_prev + (target - v_prev) / slope
                    break
            
            if pd.isna(hl) and ic_vals.iloc[-1] > target:
                hl = np.inf # Has not decayed by half even at max horizon
                
            half_lives[feature] = hl
            
        self.half_lives = half_lives
        return half_lives

    def suggest_optimal_holding_period(self) -> str:
        """Suggest optimal portfolio holding period by aggregating signal half-lives."""
        if not self.half_lives:
            self.compute_decay_half_life()
            
        valid_hls = [v for v in self.half_lives.values() if not pd.isna(v) and v != np.inf and v > 0]
        
        if not valid_hls:
            return "Unable to determine holding period (insufficient decay data)."
            
        median_hl = np.median(valid_hls)
        
        # Typically, a strategy executes well capturing signal before full decay
        optimal_days = max(1, int(round(median_hl)))
        
        suggestion = (
            f"Suggested Optimal Holding Period: {optimal_days} days\n"
            f"(Derived from median signal half-life of {median_hl:.1f} days across valid features)"
        )
        logger.info(suggestion)
        return suggestion

    def plot_decay_curve(self, top_n: int = 5, save_path: str = None) -> plt.Figure:
        """Plot Absolute IC vs Horizon for the strongest signals to visualize alpha decay."""
        if self.ic_results is None:
            raise ValueError("Must compute IC first.")
            
        top_features = self.ic_results['IC(1d)'].abs().nlargest(top_n).index
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for feature in top_features:
            y_vals = self.ic_results.loc[feature].abs().values
            ax.plot(self.horizons, y_vals, marker='o', linewidth=2, label=feature)
            
        ax.set_title('Signal Decay: Absolute Information Coefficient vs Horizon', fontsize=14, pad=15)
        ax.set_xlabel('Horizon (Trading Days)', fontsize=12)
        ax.set_ylabel('| Information Coefficient (Rank IC) |', fontsize=12)
        ax.set_xticks(self.horizons)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title='Top Features by 1D IC', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved decay plot to: {save_path}")
            
        return fig

    def get_summary_table(self) -> pd.DataFrame:
        """Return formatted summary table descending by predictive strength."""
        if self.ic_results is None:
            raise ValueError("Must compute IC first.")
            
        df_out = self.ic_results.copy()
        
        # Add half-lives
        if not self.half_lives:
            self.compute_decay_half_life()
        df_out['Half_Life'] = pd.Series(self.half_lives)
        
        # Sort by 1-day predictive strength (magnitude)
        df_out['|IC(1d)|'] = df_out['IC(1d)'].abs()
        df_out = df_out.sort_values('|IC(1d)|', ascending=False)
        df_out = df_out.drop(columns=['|IC(1d)|'])
        
        return df_out
