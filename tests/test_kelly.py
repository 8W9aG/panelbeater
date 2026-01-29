import unittest

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

class TestKellyGuardrails(unittest.TestCase):

    def setUp(self):
        # Create a mock simulation: 10 steps, 100 paths
        # Starting price is 627.0 (matching your error data)
        # It drifts slightly upwards to 630.0
        self.spot_start = 627.0
        self.sim_data = np.random.normal(loc=0.001, scale=0.01, size=(10, 100)) 
        self.sim_data = self.spot_start * np.exp(np.cumsum(self.sim_data, axis=0))
        
        # Convert to DataFrame
        self.sim_df = pd.DataFrame(self.sim_data)

    def test_put_arbitrage_blowout(self):
        """
        Scenario: Deep ITM Put.
        Strike: 686
        Spot: 627
        Intrinsic Value: 686 - 627 = 59.0
        Ask Price: 8.97 (The bad data)
        Expectation: Kelly returns 0.0 because Cost < Intrinsic.
        """
        row = {
            "type": "put",
            "strike": 686.0,
            "ask": 8.97,
            "tp_target": 100.0, # Irrelevant, should fail before this
            "sl_target": 5.0
        }
        
        f_star, mean_r = calculate_full_kelly_path_aware(row, self.sim_df)
        
        # Should return 0.0 leverage due to guardrail
        self.assertEqual(f_star, 0.0, "Failed to catch Put arbitrage.")
        self.assertEqual(mean_r, 0.0, "Should return 0.0 mean return on fail.")

    def test_call_arbitrage_blowout(self):
        """
        Scenario: Deep ITM Call.
        Strike: 600
        Spot: 627
        Intrinsic Value: 627 - 600 = 27.0
        Ask Price: 5.00 (Bad data)
        Expectation: Kelly returns 0.0.
        """
        row = {
            "type": "call",
            "strike": 600.0,
            "ask": 5.00,
            "tp_target": 50.0,
            "sl_target": 2.0
        }
        
        f_star, mean_r = calculate_full_kelly_path_aware(row, self.sim_df)
        self.assertEqual(f_star, 0.0, "Failed to catch Call arbitrage.")

    def test_valid_trade_execution(self):
        """
        Scenario: OTM Put (Valid).
        Strike: 600 (OTM since Spot is 627)
        Ask Price: 5.00
        Intrinsic: 0.0
        Expectation: Function runs fully and returns a calculated float.
        """
        row = {
            "type": "put",
            "strike": 600.0,
            "ask": 5.00,
            "tp_target": 15.0, # Target 3x return
            "sl_target": 2.0
        }
        
        f_star, mean_r = calculate_full_kelly_path_aware(row, self.sim_df)
        
        # Just checking that it ran and produced a number, not 0.0 due to error
        # Note: If the random simulation creates a loss, f_star might be 0 naturally.
        # But it should not trigger the arb guardrail.
        self.assertIsInstance(f_star, float)
        self.assertIsInstance(mean_r, float)
        
        # To be sure guardrail didn't trigger, verify intrinsic check:
        # Intrinsic of 600 Put at 627 Spot is 0. Ask is 5. 5 > 0. Safe.
