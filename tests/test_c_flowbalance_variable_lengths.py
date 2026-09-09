import importlib.util
import os
import torch
import pytest

module_path = os.path.abspath("verl/verl/trainer/ppo/c_flowbalance_adv.py")
spec = importlib.util.spec_from_file_location("c_flowbalance_adv", module_path)
c_flowbalance_adv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c_flowbalance_adv)
compute_c_flowbalance_advantage = c_flowbalance_adv.compute_c_flowbalance_advantage

def test_flow_gae_variable_length_bootstrap():
    """Verify that Flow-GAE correctly bootstraps the terminal token of variable-length sequences."""
    batch_size = 3
    seq_dim = 10
    
    # Sequence 0: length 4 (padded 4..9)
    # Sequence 1: length 7 (padded 7..9)
    # Sequence 2: length 10 (no padding)
    response_mask = torch.zeros(batch_size, seq_dim)
    response_mask[0, :4] = 1.0
    response_mask[1, :7] = 1.0
    response_mask[2, :10] = 1.0
    
    old_log_prob = torch.full((batch_size, seq_dim), -1.0)
    ref_log_prob = torch.full((batch_size, seq_dim), -1.0)
    teacher_log_prob = torch.full((batch_size, seq_dim), -0.8)
    
    token_level_rewards = torch.zeros(batch_size, seq_dim)
    token_level_rewards[0, 3] = 1.0
    token_level_rewards[1, 6] = 0.0
    token_level_rewards[2, 9] = 1.0
    uid_list = ["prompt_0", "prompt_0", "prompt_0"]
    
    subtb_lambda = 0.6
    
    # 1. Run with flow_gae_mode = True
    adv_gae, _, metrics = compute_c_flowbalance_advantage(
        token_level_rewards=token_level_rewards,
        ref_log_prob=ref_log_prob,
        old_log_prob=old_log_prob,
        response_mask=response_mask,
        index=uid_list,
        teacher_log_prob=teacher_log_prob,
        subtb_lambda=subtb_lambda,
        flow_gae_mode=True,
    )
    
    # 2. Run with standard 2-point SubTB for comparison
    adv_subtb, _, _ = compute_c_flowbalance_advantage(
        token_level_rewards=token_level_rewards,
        ref_log_prob=ref_log_prob,
        old_log_prob=old_log_prob,
        response_mask=response_mask,
        index=uid_list,
        teacher_log_prob=teacher_log_prob,
        subtb_lambda=subtb_lambda,
        flow_gae_mode=False,
    )
    
    # Check padding is 0
    assert torch.all(adv_gae[0, 4:] == 0.0)
    assert torch.all(adv_gae[1, 7:] == 0.0)
    
    # Terminal tokens: index 3 for seq 0, index 6 for seq 1, index 9 for seq 2
    # At terminal tokens, Flow-GAE and 2-point SubTB must be IDENTICAL:
    # A_GAE[terminal] = (1 - lambda) * A_DB[terminal] + lambda * A_TB[terminal] == A_SubTB[terminal]
    assert torch.isclose(adv_gae[0, 3], adv_subtb[0, 3], atol=1e-5), f"Seq 0 terminal mismatch: GAE={adv_gae[0, 3]}, SubTB={adv_subtb[0, 3]}"
    assert torch.isclose(adv_gae[1, 6], adv_subtb[1, 6], atol=1e-5), f"Seq 1 terminal mismatch: GAE={adv_gae[1, 6]}, SubTB={adv_subtb[1, 6]}"
    assert torch.isclose(adv_gae[2, 9], adv_subtb[2, 9], atol=1e-5), f"Seq 2 terminal mismatch: GAE={adv_gae[2, 9]}, SubTB={adv_subtb[2, 9]}"
    
    print("test_flow_gae_variable_length_bootstrap PASSED!")

if __name__ == "__main__":
    test_flow_gae_variable_length_bootstrap()
