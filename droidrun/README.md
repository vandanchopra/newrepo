<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="./static/droidrun-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/droidrun.png">
  <img src="./static/droidrun.png"  width="full">
</picture>

<div align="center">

[![Docs](https://img.shields.io/badge/Docs-📕-0D9373?style=for-the-badge)](https://docs.droidrun.ai)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-0D9373?style=for-the-badge)](https://cloud.droidrun.ai/sign-in?waitlist=true)


[![GitHub stars](https://img.shields.io/github/stars/droidrun/droidrun?style=social)](https://github.com/droidrun/droidrun/stargazers)
[![droidrun.ai](https://img.shields.io/badge/droidrun.ai-white)](https://droidrun.ai)
[![Twitter Follow](https://img.shields.io/twitter/follow/droid_run?style=social)](https://x.com/droid_run)
[![Discord](https://img.shields.io/discord/1360219330318696488?color=white&label=Discord&logo=discord&logoColor=white)](https://discord.gg/ZZbKEZZkwK)
[![Benchmark](https://img.shields.io/badge/Benchmark-91.4﹪-white)](https://droidrun.ai/benchmark)



<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=983810&theme=dark&period=daily&t=1753948032207">
  <source media="(prefers-color-scheme: light)" srcset="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=983810&theme=neutral&period=daily&t=1753948125523">
  <a href="https://www.producthunt.com/products/droidrun-framework-for-mobile-agent?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-droidrun" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=983810&theme=neutral&period=daily&t=1753948125523" alt="Droidrun - Give&#0032;AI&#0032;native&#0032;control&#0032;of&#0032;physical&#0032;&#0038;&#0032;virtual&#0032;phones&#0046; | Product Hunt" style="width: 200px; height: 54px;" width="200" height="54" /></a>
</picture>


[Deutsch](https://zdoc.app/de/droidrun/droidrun) | 
[Español](https://zdoc.app/es/droidrun/droidrun) | 
[français](https://zdoc.app/fr/droidrun/droidrun) | 
[日本語](https://zdoc.app/ja/droidrun/droidrun) | 
[한국어](https://zdoc.app/ko/droidrun/droidrun) | 
[Português](https://zdoc.app/pt/droidrun/droidrun) | 
[Русский](https://zdoc.app/ru/droidrun/droidrun) | 
[中文](https://zdoc.app/zh/droidrun/droidrun)

</div>



DroidRun is a powerful framework for controlling Android and iOS devices through LLM agents. It allows you to automate device interactions using natural language commands. [Checkout our benchmark results](https://droidrun.ai/benchmark)

## Why Droidrun?

- 🤖 Control Android and iOS devices with natural language commands
- 🔀 Supports multiple LLM providers (OpenAI, Anthropic, Gemini, Ollama, DeepSeek)
- 🧠 Planning capabilities for complex multi-step tasks
- 💻 Easy to use CLI with enhanced debugging features
- 🐍 Extendable Python API for custom automations
- 📸 Screenshot analysis for visual understanding of the device
- 🫆 Execution tracing with Arize Phoenix

## 📦 Installation

```bash
pip install 'droidrun[google,anthropic,openai,deepseek,ollama,dev]'
```

## 🚀 Quickstart
Read on how to get droidrun up and running within seconds in [our docs](https://docs.droidrun.ai/v3/quickstart)!   

[![Quickstart Video](https://img.youtube.com/vi/4WT7FXJah2I/0.jpg)](https://www.youtube.com/watch?v=4WT7FXJah2I)

## 🎬 Demo Videos

1. **Accommodation booking**: Let Droidrun search for an apartment for you

   [![Droidrun Accommodation Booking Demo](https://img.youtube.com/vi/VUpCyq1PSXw/0.jpg)](https://youtu.be/VUpCyq1PSXw)

<br>

2. **Trend Hunter**: Let Droidrun hunt down trending posts

   [![Droidrun Trend Hunter Demo](https://img.youtube.com/vi/7V8S2f8PnkQ/0.jpg)](https://youtu.be/7V8S2f8PnkQ)

<br>

3. **Streak Saver**: Let Droidrun save your streak on your favorite language learning app

   [![Droidrun Streak Saver Demo](https://img.youtube.com/vi/B5q2B467HKw/0.jpg)](https://youtu.be/B5q2B467HKw)


## 💡 Example Use Cases

- Automated UI testing of mobile applications
- Creating guided workflows for non-technical users
- Automating repetitive tasks on mobile devices
- Remote assistance for less technical users
- Exploring mobile UI with natural language commands

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 

## Security Checks

To ensure the security of the codebase, we have integrated security checks using `bandit` and `safety`. These tools help identify potential security issues in the code and dependencies.

### Running Security Checks

Before submitting any code, please run the following security checks:

1. **Bandit**: A tool to find common security issues in Python code.
   ```bash
   bandit -r droidrun
   ```

2. **Safety**: A tool to check your installed dependencies for known security vulnerabilities.
   ```bash
   safety scan
   ```

---

## Extended Features (v2.0)

The following features have been added for autonomous long-running operation:

### Memory System
Episodic memory with vector similarity search for recalling past interactions.

```python
from droidrun.agent.memory import MemoryManager, MemoryConfig

config = MemoryConfig(
    store_type="in_memory",  # or "qdrant" for production
    similarity_threshold=0.7,
    max_recalled_episodes=5,
)
memory = MemoryManager(config=config)
```

**Features:**
- Local embeddings (sentence-transformers) - no external API required
- In-memory store for testing, Qdrant for production
- Similarity-based episode recall

### Research Agent
Multi-provider search for deep research capabilities.

```python
from droidrun.agent.research import ResearchAgent, ResearchAgentConfig

config = ResearchAgentConfig(offline_mode=True)  # Set False with API keys
agent = ResearchAgent(config=config)
result = await agent.research("latest Android automation techniques")
```

**Providers:** Tavily, Brave Search, Mock (for testing)

### Claude Agent Integration
Claude Code agent wrapper with streaming and offline mode.

```python
from droidrun.agent.claude import ClaudeCodeAgent, ClaudeAgentConfig

config = ClaudeAgentConfig(offline_mode=True)  # Set False with API key
agent = ClaudeCodeAgent(config=config)
response = await agent.complete("Analyze this code...")
```

### State Checkpointing
Automatic state persistence for long-running tasks.

```python
from droidrun.agent.checkpoint import CheckpointManager, CheckpointConfig

config = CheckpointConfig(
    checkpoint_dir="checkpoints",
    interval_seconds=300,  # Save every 5 minutes
    max_checkpoints=10,
)
manager = CheckpointManager(config=config)
```

**Features:**
- Periodic auto-save
- Checkpoint rotation (keep N most recent)
- Integrity validation
- Resume from any checkpoint

### Task Scheduler
Priority-based task scheduling with persistence.

```python
from droidrun.scheduler import TaskScheduler, SchedulerConfig, TaskPriority

config = SchedulerConfig(storage_path="tasks.json")
scheduler = TaskScheduler(config=config)

# Schedule a high-priority task
task_id = scheduler.schedule_task(
    "Open YouTube and search for tutorials",
    priority=TaskPriority.HIGH,
    max_retries=3,
)
```

**Features:**
- Priority levels: HIGH, NORMAL, LOW
- Delayed execution
- Cron expression support
- Task persistence across restarts

### Web Interface

**Backend (FastAPI):**
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Frontend (React + Vite):**
```bash
cd frontend && npm run dev
```

**Docker:**
```bash
docker compose up -d agent frontend qdrant redis
```

### Test Suite

```bash
# Run all tests (107 tests)
python -m pytest tests/ -v

# Results:
# - test_memory.py: 51 passed
# - test_research.py: 9 passed
# - test_claude_agent.py: 16 passed
# - test_checkpoint.py: 15 passed
# - test_scheduler.py: 16 passed
# Total: 107 passed, 1 skipped
```

### Configuration

Add to your environment:

```bash
# For production memory
MEMORY_STORE=qdrant
QDRANT_URL=http://localhost:6333

# For research
TAVILY_API_KEY=your_key
BRAVE_SEARCH_API_KEY=your_key

# For Claude agent
ANTHROPIC_API_KEY=your_key

# Development mode (no API keys needed)
OFFLINE_MODE=true
```
