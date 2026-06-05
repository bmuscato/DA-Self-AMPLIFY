# DA-Self-AMPLIFY: Improving In-Context Learning with Post-hoc Explanations for Subjective NLP

# Abstract


In-context learning (ICL) is a cost-effective, low-resource adaptation strategy for aligning Large Language
Models (LLMs), requiring substantially less annotated data than fine-tuning. Although ICL is highly sensitive
to demonstration selection and ordering, recent studies suggest that leveraging annotator disagreement can
improve performance on subjective NLP tasks. Furthermore, human explanations—highlighted tokens or free-text
rationales—can serve as an additional learning signal. Yet their use in subjective NLP remains underexplored,
largely due to the scarcity of datasets with human-provided explanations. To address these limitations, we
introduce DA-Self-Amplify, a scalable framework that enriches ICL with disagreement-aware demonstrations
and automatically generated post-hoc explanations. Specifically, demonstrations are selected based on annotator
disagreement and grouped into easy, ambiguous, and difficult, and augmented with either feature-attribution
or free-text explanations. We evaluate DA-Self-Amplify on three subjective classification tasks in English and
Italian using Llama-3-8B-Instruct and its Italian-adapted variant, LLaMAntino-3-ANITA-8B-Inst-DPO-ITA.
Results show that CoT-based approaches consistently outperform standard prompting and that explanations
provide an effective learning signal. This is particularly pronounced for ambiguous and difficult demonstration
examples, highlighting the importance of incorporating annotator disagreement into ICL for subjective NLP
tasks.

