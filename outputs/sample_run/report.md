# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_mini.json
- Mode: real
- Records: 16
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.75 | 1.0 | 0.25 |
| Avg attempts | 1 | 1.125 | 0.125 |
| Avg token estimate | 184 | 217 | 33 |
| Avg latency (ms) | 3867.75 | 3945.25 | 77.5 |

## Failure modes
```json
{
  "react": {
    "none": 6,
    "entity_drift": 2
  },
  "reflexion": {
    "none": 8
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
