import numpy as np
import pandas as pd

from panelbeater.kelly import calculate_full_kelly_path_aware


class TestKellySafeguards:
    
    def setup_method(self):
        """Create a mock simulation matrix (Time x Paths)."""
        # FIX: Make the drift stronger (100 -> 130) so it clears the $115 target easily.
        self.sim_data = np.linspace(100, 130, 10).reshape(-1, 1) + np.random.normal(0, 1, (10, 5))
        self.sim_df = pd.DataFrame(self.sim_data) # Wide format

    def test_negative_edge_returns_zero_kelly(self):
        """
        CRITICAL TEST: Matches your SPY scenario.
        Ask is 16.91, Target is 16.21. 
        Kelly MUST be 0.
        """
        bad_trade_row = pd.Series({
            "ask": 16.91,
            "strike": 675,
            "type": "call",
            "tp_target": 16.21,  # <--- The Trap
            "sl_target": 5.00
        })

        f_star, expected_val = calculate_full_kelly_path_aware(bad_trade_row, self.sim_df)

        assert f_star == 0.0, f"Kelly should be 0 for negative edge, got {f_star}"
        assert expected_val == 0.0, "Expected value should be 0 (or negative) for guaranteed loss"

    def test_positive_edge_allows_calculation(self):
        """
        Verify we didn't break valid trades.
        We need diverse outcomes (wins AND losses) to generate variance 
        so Kelly calculation works.
        """
        # 1. Create Mixed Outcomes
        # 4 Paths go to 140 (Winners, hit TP)
        winners = np.linspace(100, 140, 20).reshape(-1, 1) 
        winners = np.tile(winners, (1, 4)) # 4 columns
        
        # 1 Path goes to 90 (Loser, hits SL)
        loser = np.linspace(100, 90, 20).reshape(-1, 1)
        
        # Combine them: 5 columns total
        mixed_sim_data = np.hstack([winners, loser])
        
        # Add slight noise to ensure no numerical weirdness
        mixed_sim_data += np.random.normal(0, 0.5, mixed_sim_data.shape)
        
        mixed_sim_df = pd.DataFrame(mixed_sim_data)

        good_trade_row = pd.Series({
            "ask": 10.00,
            "strike": 100,
            "type": "call",
            "tp_target": 15.00, # Requires stock > 115
            "sl_target": 5.00   # Requires stock < ~95
        })

        # 2. Calculate
        f_star, _ = calculate_full_kelly_path_aware(good_trade_row, mixed_sim_df)
        
        # Now we have Mean > 0 (mostly winners) AND Variance > 0 (mixed results)
        assert f_star > 0.0, f"Valid trade filtered out. Kelly was {f_star}"
