import numpy as np
import coremltools as ct
import time
import os

def validate():
    # a) Load golden reference data
    golden_moe = np.load('python/privacy/out/pf_alllayers_moe.npz')
    golden_weights = golden_moe['L0_topk_weights'][0] # Sentence 0
    golden_indices = golden_moe['L0_topk_indices'][0]
    
    # b) Get attn output
    attn_data = np.load('python/privacy/out/pf_attn_alllayers.npz')
    attn_out_s0 = attn_data['L0_attn_out'][0] # [128, 640]

    # Model path
    model_path = os.path.abspath('emilio/conv-ane/PF_router_L0_T128.mlpackage')
    
    def run_benchmark(compute_unit, iterations=100, warmup=10):
        config = ct.ComputeUnit[compute_unit]
        mlmodel = ct.models.MLModel(model_path, compute_units=config)
        
        # d) Pack input [1, 640, 1, 128]
        input_data = attn_out_s0.T.reshape(1, 640, 1, 128).astype(np.float16)
        
        # Warmup
        for _ in range(warmup):
            _ = mlmodel.predict({'x_in': input_data})
            
        start = time.perf_counter()
        for _ in range(iterations):
            out = mlmodel.predict({'x_in': input_data})
        end = time.perf_counter()
        
        return out, (end - start) / iterations

    try:
        if not os.path.exists(model_path):
             print(f"Error: {model_path} does not exist")
             return

        print("Running on CPU Only...")
        out_cpu, time_cpu = run_benchmark('CPU_ONLY')
        print(f"CPU Time: {time_cpu*1000:.3f}ms")

        print("Running on CPU and Neural Engine...")
        out_ane, time_ane = run_benchmark('CPU_AND_NE')
        print(f"ANE/CPU Time: {time_ane*1000:.3f}ms")

        # e/f) Output processing
        # Use first key by default
        probs_key = list(out_ane.keys())[0]
        print(f"Using output key: {probs_key}")
            
        probs = out_ane[probs_key] # shape (1, 128, 1, 128) -> [experts, tokens]
        
        # Reshape: [1, N_EXPERTS=128, 1, T=128] -> [T, N_EXPERTS]
        probs = probs.reshape(128, 128).T  # now [T=128, E=128]

        # g) Compare top-4 indices
        top4_indices_pred = np.zeros((128, 4))
        for i in range(128):
            top4_indices_pred[i] = np.argsort(probs[i])[-4:][::-1]
        
        match_count = 0
        for i in range(128):
            if set(top4_indices_pred[i].astype(int)) == set(golden_indices[i].astype(int)):
                match_count += 1
        
        match_rate = match_count / 128.0

        # h) Cosine similarity
        mean_cos_sim = 0
        for i in range(128):
            pred_vec = probs[i].astype(np.float32)
            gold_vec = np.zeros(128)
            gold_vec[golden_indices[i].astype(int)] = golden_weights[i] * 4.0
            
            num = np.dot(pred_vec, gold_vec)
            den = np.linalg.norm(pred_vec) * np.linalg.norm(gold_vec)
            if den > 1e-9:
                mean_cos_sim += num / den
        
        mean_cos_sim /= 128.0

        print(f"Top-4 Match Rate: {match_rate:.4f}")
        print(f"Mean Cosine Similarity: {mean_cos_sim:.4f}")
        print(f"ANE Speedup: {time_cpu/time_ane:.2f}x")
        if time_ane < 0.7 * time_cpu:
            print("ANE Residency: Likely")
        else:
            print("ANE Residency: Unlikely")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    validate()
