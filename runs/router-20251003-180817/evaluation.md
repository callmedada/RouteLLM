# Evaluation Report

- avg_cost: 0.00018503799219615757
- top1_acc: 0.15919409761634506
- cost_per_correct: 0.0011623420401056315

## Route Ratios

- gpt-3.5-turbo-1106: 0.0
- claude-instant-v1: 0.0
- claude-v1: 0.0
- claude-v2: 0.0
- gpt-4-1106-preview: 0.0
- meta/llama-2-70b-chat: 0.0
- mistralai/mixtral-8x7b-chat: 0.009506242905788876
- zero-one-ai/Yi-34B-Chat: 0.9904937570942112
- WizardLM/WizardLM-13B-V1.2: 0.0
- meta/code-llama-instruct-34b-chat: 0.0
- mistralai/mistral-7b-chat: 0.0

## Per-model (precision/recall/f1)

- gpt-3.5-turbo-1106: precision=0.0000, recall=0.0000, f1=0.0000, support=236, cost=0.00024326004495378584
- claude-instant-v1: precision=0.0000, recall=0.0000, f1=0.0000, support=340, cost=0.00023293713456951082
- claude-v1: precision=0.0000, recall=0.0000, f1=0.0000, support=91, cost=0.0021446554455906153
- claude-v2: precision=0.0000, recall=0.0000, f1=0.0000, support=68, cost=0.0024186039809137583
- gpt-4-1106-preview: precision=0.0000, recall=0.0000, f1=0.0000, support=167, cost=0.0032927528955042362
- meta/llama-2-70b-chat: precision=0.0000, recall=0.0000, f1=0.0000, support=134, cost=0.0002026650181505829
- mistralai/mixtral-8x7b-chat: precision=0.3731, recall=0.0202, f1=0.0383, support=1239, cost=0.00013457989552989602
- zero-one-ai/Yi-34B-Chat: precision=0.1571, recall=0.9991, f1=0.2716, support=1098, cost=0.00018552225083112717
- WizardLM/WizardLM-13B-V1.2: precision=0.0000, recall=0.0000, f1=0.0000, support=1667, cost=7.291440124390647e-05
- meta/code-llama-instruct-34b-chat: precision=0.0000, recall=0.0000, f1=0.0000, support=96, cost=0.00017204870528075844
- mistralai/mistral-7b-chat: precision=0.0000, recall=0.0000, f1=0.0000, support=1912, cost=4.572428952087648e-05

## Calibration

- ECE: 0.005255293318838997  MCE: 0.14543871581554413
- After T=0.751, ECE: 0.029988114596275817  MCE: 0.09182757139205933

## Baselines

- Best fixed: mistralai/mistral-7b-chat (acc=0.27128263337116915, cost=4.572428952087648e-05)
- Oracle cheapest-correct cost: 0.0002368361669570905

## Frontier

- Best under current acc: tau=0.3, acc=0.27128263337116915, cost=4.572428952087648e-05, savings_vs_current=0.752892425073417

## Subgroups by eval

- Chinese_character_riddles: top1_acc=0.6, avg_cost=0.00018552225083112717
- abstract2title: top1_acc=0.0, avg_cost=0.00018552225083112717
- accounting_audit: top1_acc=0.0, avg_cost=0.00018552225083112717
- arc-challenge: top1_acc=0.043478260869565216, avg_cost=0.00018552225083112717
- bias_detection: top1_acc=0.0, avg_cost=0.00018552225083112717
- chinese-lantern-riddles: top1_acc=1.0, avg_cost=0.00018552225083112717
- chinese_ancient_masterpieces_dynasty: top1_acc=1.0, avg_cost=0.00018552225083112717
- chinese_chu_ci: top1_acc=0.0, avg_cost=0.00018552225083112717
- chinese_famous_novel: top1_acc=0.0, avg_cost=0.00018552225083112717
- chinese_hard_translations: top1_acc=0.0, avg_cost=0.00018552225083112717
- chinese_homonym: top1_acc=0.0, avg_cost=0.00018552225083112717
- chinese_idioms: top1_acc=0.0, avg_cost=0.00018552225083112717
- chinese_modern_poem_identification: top1_acc=0.0, avg_cost=0.00018552225083112717
- chinese_poem: top1_acc=0.0, avg_cost=0.00018552225083112717
- chinese_shi_jing: top1_acc=0.0, avg_cost=0.00018552225083112717
- chinese_song_ci: top1_acc=0.0, avg_cost=0.00018552225083112717
- chinese_tang_poetries: top1_acc=0.0, avg_cost=0.00018552225083112717
- chinese_zodiac: top1_acc=0.373134328358209, avg_cost=0.00013457989552989602
- consensus_summary: top1_acc=0.08974358974358974, avg_cost=0.00018552225083112717
- grade-school-math: top1_acc=0.09410205434062292, avg_cost=0.00018552223627921194
- hellaswag: top1_acc=0.34555499748869917, avg_cost=0.00018552225083112717
- mbpp: top1_acc=0.0136986301369863, avg_cost=0.00018552225083112717
- mmlu-abstract-algebra: top1_acc=0.3125, avg_cost=0.00018552225083112717
- mmlu-anatomy: top1_acc=0.16666666666666666, avg_cost=0.00018552225083112717
- mmlu-astronomy: top1_acc=0.12121212121212122, avg_cost=0.00018552225083112717
- mmlu-business-ethics: top1_acc=0.0, avg_cost=0.00018552225083112717
- mmlu-clinical-knowledge: top1_acc=0.07575757575757576, avg_cost=0.00018552225083112717
- mmlu-college-biology: top1_acc=0.0, avg_cost=0.00018552225083112717
- mmlu-college-chemistry: top1_acc=0.2631578947368421, avg_cost=0.00018552225083112717
- mmlu-college-computer-science: top1_acc=0.15, avg_cost=0.00018552225083112717
- mmlu-college-mathematics: top1_acc=0.1875, avg_cost=0.00018552225083112717
- mmlu-college-medicine: top1_acc=0.08108108108108109, avg_cost=0.00018552225083112717
- mmlu-college-physics: top1_acc=0.12, avg_cost=0.00018552225083112717
- mmlu-computer-security: top1_acc=0.09523809523809523, avg_cost=0.00018552225083112717
- mmlu-conceptual-physics: top1_acc=0.1111111111111111, avg_cost=0.00018552225083112717
- mmlu-econometrics: top1_acc=0.18518518518518517, avg_cost=0.00018552225083112717
- mmlu-electrical-engineering: top1_acc=0.08333333333333333, avg_cost=0.00018552225083112717
- mmlu-elementary-mathematics: top1_acc=0.21428571428571427, avg_cost=0.00018552225083112717
- mmlu-formal-logic: top1_acc=0.0, avg_cost=0.00018552225083112717
- mmlu-global-facts: top1_acc=0.05263157894736842, avg_cost=0.00018552225083112717
- mmlu-high-school-biology: top1_acc=0.04838709677419355, avg_cost=0.00018552225083112717
- mmlu-high-school-chemistry: top1_acc=0.05128205128205128, avg_cost=0.00018552225083112717
- mmlu-high-school-computer-science: top1_acc=0.045454545454545456, avg_cost=0.00018552225083112717
- mmlu-high-school-european-history: top1_acc=0.02631578947368421, avg_cost=0.00018552225083112717
- mmlu-high-school-geography: top1_acc=0.0, avg_cost=0.00018552225083112717
- mmlu-high-school-government-and-politics: top1_acc=0.05128205128205128, avg_cost=0.00018552225083112717
- mmlu-high-school-macroeconomics: top1_acc=0.05, avg_cost=0.00018552225083112717
- mmlu-high-school-mathematics: top1_acc=0.32608695652173914, avg_cost=0.00018552225083112717
- mmlu-high-school-microeconomics: top1_acc=0.023255813953488372, avg_cost=0.00018552225083112717
- mmlu-high-school-physics: top1_acc=0.14285714285714285, avg_cost=0.00018552225083112717
- mmlu-high-school-psychology: top1_acc=0.10434782608695652, avg_cost=0.00018552227993495762
- mmlu-high-school-statistics: top1_acc=0.09375, avg_cost=0.00018552225083112717
- mmlu-high-school-us-history: top1_acc=0.10638297872340426, avg_cost=0.00018552225083112717
- mmlu-high-school-world-history: top1_acc=0.0, avg_cost=0.00018552225083112717
- mmlu-human-aging: top1_acc=0.09523809523809523, avg_cost=0.00018552225083112717
- mmlu-human-sexuality: top1_acc=0.16, avg_cost=0.00018552225083112717
- mmlu-international-law: top1_acc=0.0, avg_cost=0.00018552225083112717
- mmlu-jurisprudence: top1_acc=0.058823529411764705, avg_cost=0.00018552225083112717
- mmlu-logical-fallacies: top1_acc=0.0, avg_cost=0.00018552225083112717
- mmlu-machine-learning: top1_acc=0.15384615384615385, avg_cost=0.00018552225083112717
- mmlu-management: top1_acc=0.08695652173913043, avg_cost=0.00018552225083112717
- mmlu-marketing: top1_acc=0.0, avg_cost=0.00018552225083112717
- mmlu-medical-genetics: top1_acc=0.047619047619047616, avg_cost=0.00018552225083112717
- mmlu-miscellaneous: top1_acc=0.0196078431372549, avg_cost=0.00018552225083112717
- mmlu-moral-disputes: top1_acc=0.03333333333333333, avg_cost=0.00018552225083112717
- mmlu-moral-scenarios: top1_acc=0.21910112359550563, avg_cost=0.0001855222653830424
- mmlu-nutrition: top1_acc=0.0196078431372549, avg_cost=0.00018552225083112717
- mmlu-philosophy: top1_acc=0.06896551724137931, avg_cost=0.00018552225083112717
- mmlu-prehistory: top1_acc=0.0625, avg_cost=0.00018552225083112717
- mmlu-professional-accounting: top1_acc=0.12, avg_cost=0.00018552225083112717
- mmlu-professional-law: top1_acc=0.07352941176470588, avg_cost=0.00018552225083112717
- mmlu-professional-medicine: top1_acc=0.046511627906976744, avg_cost=0.00018552225083112717
- mmlu-professional-psychology: top1_acc=0.09009009009009009, avg_cost=0.00018552230903878808
- mmlu-public-relations: top1_acc=0.05, avg_cost=0.00018552225083112717
- mmlu-security-studies: top1_acc=0.05660377358490566, avg_cost=0.00018552225083112717
- mmlu-sociology: top1_acc=0.10638297872340426, avg_cost=0.00018552225083112717
- mmlu-us-foreign-policy: top1_acc=0.0, avg_cost=0.00018552225083112717
- mmlu-virology: top1_acc=0.09090909090909091, avg_cost=0.00018552225083112717
- mmlu-world-religions: top1_acc=0.0, avg_cost=0.00018552225083112717
- mtbench: top1_acc=0.16666666666666666, avg_cost=0.00018552225083112717
- mtbench-math: top1_acc=0.0, avg_cost=0.00018552225083112717
- mtbench-reference: top1_acc=0.0, avg_cost=0.00018552225083112717
- test-match: top1_acc=0.0, avg_cost=0.00018552225083112717
- winogrande: top1_acc=0.015748031496062992, avg_cost=0.00018552223627921194
