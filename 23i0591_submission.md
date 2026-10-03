Full name: Abdul Rauf
Roll number: 23I-0591
Class / section: Agentic AI - Fall 2026
University email: i230591@isb.nu.edu.pk
GitHub username: rauf17
Agent name: Inventory & Purchasing Sentinel Agent
Domain: Inbox Triage & Inventory Management

GitHub repository URL: https://github.com/rauf17/stockpilot-ai
Final commit hash: bb279025547f91aed15ce00122c0379cc93f8166
Working agent interface: Pending operator deployment on Render (Blueprint in render.yaml)
Health endpoint (GET): Pending operator deployment (https://stockpilot-ai.onrender.com/health)
Arena endpoint (POST): Pending operator deployment (https://stockpilot-ai.onrender.com/arena/run)
Manifest endpoint (GET): Pending operator deployment (https://stockpilot-ai.onrender.com/arena/manifest)
API documentation: Pending operator deployment (https://stockpilot-ai.onrender.com/docs)

Hosting provider: Render Web Service (render.yaml blueprint ready)
Default model / provider: Gemini 2.5 Flash / Google GenAI (with strict LLMProviderError routing enforcement)
Other available models: openai/gpt-oss-20b (Groq Cloud Free Plan)
Example input: {"task": "Check inventory and report low stock items", "arena_config": {"fault": "none"}}
Expected result: Status completed, stop_reason goal_completed, tool_calls containing check_inventory.
Cold-start / restart limitations: In-memory SessionState resets on server restart or worker restart.
Repository access: Ready for instructor access on GitHub
Public test results: evaluation/public_cases.json (8/8 cases passed)
Model experiment file: evaluation/results/summary.json (10/10 cases benchmarked)
