"""
Minimal RunPod handler for vLLM.
Starts vLLM's built-in OpenAI server as a subprocess,
then proxies RunPod jobs to it via HTTP.
No internal vLLM API imports — works with any vLLM version.
"""
import os
import subprocess
import time
import json
import requests
import runpod

VLLM_PORT = 8000
VLLM_BASE = f"http://localhost:{VLLM_PORT}"


def start_vllm_server():
    """Start vLLM OpenAI-compatible server as a subprocess."""
    model = os.environ.get("MODEL_NAME", "google/gemma-4-31B-it")
    cmd = ["python3", "-m", "vllm.entrypoints.openai.api_server",
           "--model", model,
           "--host", "0.0.0.0",
           "--port", str(VLLM_PORT)]

    # Add optional args from env vars
    env_to_flag = {
        "TENSOR_PARALLEL_SIZE": "--tensor-parallel-size",
        "MAX_MODEL_LEN": "--max-model-len",
        "GPU_MEMORY_UTILIZATION": "--gpu-memory-utilization",
        "DTYPE": "--dtype",
        "QUANTIZATION": "--quantization",
        "MAX_NUM_SEQS": "--max-num-seqs",
        "LIMIT_MM_PER_PROMPT": "--limit-mm-per-prompt",
    }
    for env_key, flag in env_to_flag.items():
        val = os.environ.get(env_key)
        if val:
            cmd.extend([flag, val])

    print(f"[handler] Starting vLLM: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Wait for server to be ready
    for i in range(300):  # 5 min max
        try:
            r = requests.get(f"{VLLM_BASE}/health", timeout=2)
            if r.status_code == 200:
                print(f"[handler] vLLM server ready after {i}s")
                return proc
        except Exception:
            pass
        time.sleep(1)

    raise RuntimeError("vLLM server failed to start within 5 minutes")


def handler(job):
    """RunPod handler — proxy request to local vLLM server."""
    job_input = job["input"]

    # Support openai_route format (standard RunPod vLLM worker format)
    openai_route = job_input.get("openai_route")
    openai_input = job_input.get("openai_input")

    if openai_route and openai_input:
        url = f"{VLLM_BASE}{openai_route}"
        resp = requests.post(url, json=openai_input, timeout=300)
        return resp.json()

    # Support direct method/url/body format
    method = job_input.get("method", "POST").upper()
    path = job_input.get("url", "/v1/chat/completions")
    body = job_input.get("body", {})

    url = f"{VLLM_BASE}{path}"
    if method == "GET":
        resp = requests.get(url, timeout=60)
    else:
        resp = requests.post(url, json=body, timeout=300)

    return resp.json()


if __name__ == "__main__":
    print("[handler] Initializing vLLM server...")
    vllm_proc = start_vllm_server()
    print("[handler] Starting RunPod handler...")
    runpod.serverless.start({"handler": handler})
