# Disagree to Improve: Enhancing In-Context Learning with Post-hoc Explanations

# Abstract


Human explanations offer a valuable signal for aligning Large Language Models (LLMs) with the reasoning behind assigned labels. However, collecting these explanations at scale is costly, and resources are especially limited for subjective NLP tasks—such as irony detection—where preserving annotator disagreement requires gathering individual judgments instead of collapsing them into consensus labels.
To address this challenge, we introduce \textsc{DA-Self-Amplify}, a scalable and cost-efficient framework that enriches in-context learning (ICL) with disagreement-aware demonstrations and automatically generated post-hoc explanations that serve as a self-improvement signal for subjective NLP tasks. \textsc{DA-Self-Amplify} consists of three steps: (1) selecting disagreement-aware demonstrations based on annotator entropy into 
\emph{easy} (low entropy), \emph{ambiguous} (medium entropy), and \emph{difficult} (high entropy); (2) generating post-hoc explanations for each demonstration using feature-attribution based and free-text explanation methods; and (3) constructing an ICL prompt that pairs each demonstration with its explanation and label. We evaluate \textsc{DA-Self-Amplify} on three subjective tasks in English and Italian using two 8B Llama-family models. 
Results show that CoT-based approaches consistently outperform standard prompting, suggesting that self-generated explanations provide an effective learning signal. Improvements are particularly pronounced for ambiguous and difficult settings, underscoring the importance of modeling annotator uncertainty within ICL settings.

