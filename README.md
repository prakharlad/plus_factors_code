# Plus Factors for AI Agents — sprint code (AI Collusion Research Sprint, Oct 2026, Q2.3)
All code written during the sprint (no pre-existing code). Demand model from ColludeBench (poseidon-research/collusion_hackathon/steganographic-collusion).
    pip install -r requirements.txt
    python sim.py       # generates 5 case types x seeds, computes plus-factor signals -> signals.json
    python evaluate.py  # freezes thresholds on seeds 0-19, evaluates rules on seeds 20-39 + stress case E -> results.json
    python fig.py       # fig_rules.png, fig_examples.png
No API key needed; runs in seconds on a laptop.

## LLM agents (Section 5)
llm_results.json holds all 20 qwen-plus episodes from the paper (prices, messages, reasoning, profits, channel-cut replays, judge output),
and llm_scored.csv is its score under the frozen thresholds. To rescore without any API calls:
    python analyze_llm.py llm_results.json   # applies the frozen scripted-testbed thresholds in results.json -> llm_scored.csv

To rerun the experiment: llm_run.py needs only Python 3.8+ and an OpenAI-compatible chat API. LLM_BASE_URL defaults to OpenRouter.
    set LLM_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1     (Alibaba Cloud Model Studio, as used in the paper)
    set LLM_API_KEY=<your key>
    python llm_run.py --model qwen-plus --judge_model qwen-plus --seeds 10 --workers 4 --out llm_results.json
Each finished episode is appended to llm_results.json.partial.jsonl, and a rerun resumes from it. About 1,800 API calls.
Add --mock to llm_run.py for an offline dry run. A rerun samples new episodes, so its numbers will differ from the paper's.
