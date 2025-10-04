# Evaluation Report

- avg_cost: 8.290598634630442e-05
- top1_acc: 0.39473684210526316
- cost_per_correct: 0.0002100284987439712

## Route Ratios

- gpt-3.5-turbo-1106: 0.008771929824561403
- claude-instant-v1: 0.0
- claude-v1: 0.0
- claude-v2: 0.0
- gpt-4-1106-preview: 0.0
- meta/llama-2-70b-chat: 0.0
- mistralai/mixtral-8x7b-chat: 0.31140350877192985
- zero-one-ai/Yi-34B-Chat: 0.0
- WizardLM/WizardLM-13B-V1.2: 0.18421052631578946
- meta/code-llama-instruct-34b-chat: 0.021929824561403508
- mistralai/mistral-7b-chat: 0.47368421052631576

## Per-model (precision/recall/f1)

- gpt-3.5-turbo-1106: precision=0.5000, recall=0.3333, f1=0.4000, support=3, cost=0.00024326004495378584
- claude-instant-v1: precision=0.0000, recall=0.0000, f1=0.0000, support=1, cost=0.00023293713456951082
- claude-v1: precision=0.0000, recall=0.0000, f1=0.0000, support=1, cost=0.0021446554455906153
- claude-v2: precision=0.0000, recall=0.0000, f1=0.0000, support=2, cost=0.0024186039809137583
- gpt-4-1106-preview: precision=0.0000, recall=0.0000, f1=0.0000, support=8, cost=0.0032927528955042362
- meta/llama-2-70b-chat: precision=0.0000, recall=0.0000, f1=0.0000, support=0, cost=0.0002026650181505829
- mistralai/mixtral-8x7b-chat: precision=0.0141, recall=0.5000, f1=0.0274, support=2, cost=0.00013457989552989602
- zero-one-ai/Yi-34B-Chat: precision=0.0000, recall=0.0000, f1=0.0000, support=9, cost=0.00018552225083112717
- WizardLM/WizardLM-13B-V1.2: precision=0.0000, recall=0.0000, f1=0.0000, support=1, cost=7.291440124390647e-05
- meta/code-llama-instruct-34b-chat: precision=0.2000, recall=1.0000, f1=0.3333, support=1, cost=0.00017204870528075844
- mistralai/mistral-7b-chat: precision=0.8056, recall=0.4350, f1=0.5649, support=200, cost=4.572428952087648e-05

## Calibration

- ECE: 0.17642259074930558  MCE: 0.3137231469154358
- After T=0.809, ECE: 0.13966725747051992  MCE: 0.21378469467163086

## Baselines

- Best fixed: mistralai/mistral-7b-chat (acc=0.8771929824561403, cost=4.572428952087648e-05)
- Oracle cheapest-correct cost: 0.00020006703903562627

## Frontier

- Best under current acc: tau=0.56, acc=0.8771929824561403, cost=4.5724285882897675e-05, savings_vs_current=0.4484802859481828

## Subgroups by eval

- Chinese_character_riddles: top1_acc=0.14285714285714285, avg_cost=6.434844544855878e-05
- abstract2title: top1_acc=0.43, avg_cost=8.550403435947374e-05
