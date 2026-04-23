# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot.json
- Mode: real
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.2 | 0.59 | 0.39 |
| Avg attempts | 1 | 2.27 | 1.27 |
| Avg token estimate | 1531.75 | 3570.9 | 2039.15 |
| Avg latency (ms) | 3762.07 | 9921.24 | 6159.17 |

## Failure modes
```json
{
  "react": {
    "wrong_final_answer": 80,
    "none": 20
  },
  "reflexion": {
    "wrong_final_answer": 41,
    "none": 59
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.
