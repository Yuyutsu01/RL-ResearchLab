import os
import yaml

def propagate():
    project_dir = r"c:\Users\shiva\OneDrive\Desktop\projects\RL-ResearchLab\reward-shaping-ppo"
    configs_dir = os.path.join(project_dir, "configs")
    
    environments = [
        {
            "opt_file": "acrobot_v1_optimized.yaml",
            "targets": ["acrobot_identity.yaml", "acrobot_dense.yaml", "acrobot_pbrs.yaml"]
        },
        {
            "opt_file": "mountaincar_v0_optimized.yaml",
            "targets": ["mountaincar_identity.yaml", "mountaincar_dense.yaml", "mountaincar_pbrs.yaml"]
        },
        {
            "opt_file": "lunarlander_v3_optimized.yaml",
            "targets": ["lunarlander_identity.yaml", "lunarlander_dense.yaml", "lunarlander_pbrs.yaml"]
        }
    ]
    
    for env in environments:
        opt_path = os.path.join(configs_dir, env["opt_file"])
        if not os.path.exists(opt_path):
            print(f"Optimized config not found (skipping propagation for this env): {opt_path}")
            continue
            
        with open(opt_path, "r") as f:
            opt_data = yaml.safe_load(f)
            
        opt_ppo = opt_data.get("ppo", {})
        
        for target_file in env["targets"]:
            target_path = os.path.join(configs_dir, target_file)
            if not os.path.exists(target_path):
                print(f"Target config not found: {target_path}")
                continue
                
            with open(target_path, "r") as f:
                target_data = yaml.safe_load(f)
                
            # Copy all PPO hyperparameters
            target_data["ppo"] = opt_ppo
            
            # Save updated target config
            with open(target_path, "w") as f:
                yaml.safe_dump(target_data, f, default_flow_style=False)
            print(f"Propagated optimized parameters to: {target_file}")
            
    print("Propagation finished.")

if __name__ == "__main__":
    propagate()
