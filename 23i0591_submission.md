Full name: Abdul Rauf
Roll number: 23i0591
Class / section: Agentic AI - Fall 2026
University email: i230591@isb.nu.edu.pk
GitHub username: rauf17
Agent name: Inventory & Purchasing Sentinel Agent
Domain: Inbox Triage & Inventory Management

GitHub repository URL: https://github.com/rauf17/stockpilot-ai
Final commit hash: 0feb7a1e45e261aa6ef6a73c8dd120e2e3414e4b
Working agent interface: https://YOUR-APP.onrender.com/
Health endpoint (GET): https://YOUR-APP.onrender.com/health
Arena endpoint (POST): https://YOUR-APP.onrender.com/arena/run
Manifest endpoint (GET): https://YOUR-APP.onrender.com/arena/manifest
API documentation: https://YOUR-APP.onrender.com/docs

Hosting provider: Render (Web Service)
Default model / provider: Gemini 2.5 Flash / Google GenAI (with explicit LLMProviderError handling)
Other available models: openai/gpt-oss-20b (Groq Cloud Free Plan)
Example input: {"task": "Check inventory and report low stock items", "arena_config": {"fault": "none"}}
Expected result: Status completed, stop_reason goal_completed, tool_calls containing check_inventory.
Cold-start / restart limitations: In-memory SessionState resets on server restart or worker restart.
Repository access: Instructor invited / access confirmed
Public test results: evaluation/public_cases.json (8/8 cases passed)
