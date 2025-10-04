#!/usr/bin/env python3
import os
import subprocess

# Test environment loading
mcp_server_path = "/Users/rutujanemane/Documents/SJSU/A10 hackathon/DoorDash-MCP-Server/build/index.js"
env_path = os.path.join(os.path.dirname(mcp_server_path), '.env')

print(f"Env path: {env_path}")
print(f"Exists: {os.path.exists(env_path)}")

# Load .env file
env = os.environ.copy()
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                env[key] = value
                print(f"Loaded: {key} = {value[:10]}...")

print(f"DOORDASH_DEVELOPER_ID in env: {'DOORDASH_DEVELOPER_ID' in env}")

# Test subprocess
result = subprocess.run(
    ["node", mcp_server_path],
    input='{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "clientInfo": {"name": "test", "version": "1.0.0"}}}',
    capture_output=True,
    text=True,
    cwd=os.path.dirname(mcp_server_path),
    env=env
)

print(f"Return code: {result.returncode}")
print(f"Stdout: {result.stdout}")
print(f"Stderr: {result.stderr}")
