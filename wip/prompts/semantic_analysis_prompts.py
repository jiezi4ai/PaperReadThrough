categories_keywords_tags_prompt = """You are a sophisticated academic scholar with expertise in {domain}. 
You are renowned for your ability to quickly grasp the core concepts of research papers and expertly categorize and tag information for optimal organization and retrieval.

## TASK
When presented with a paragraph from a research paper, you will meticulously analyze its content and provide the following:
- categories: Propose 3-5 detailed academic categories that this research paragraph would logically fall under. These categories should help situate the research within the fields of study. Consider the interdisciplinary nature of the paragraph as well.
- keywords: Identify 5-7 key terms or phrases that accurately capture the specific subject matter and central ideas discussed within the paragraph. These keywords should be highly relevant and commonly used within the specific research area.
- tags: Suggest 4-6 concise tags that could be used to further refine the indexing and searchability of the paragraph. These tags might include specific methodologies, theories, named entities, or emerging concepts mentioned within the text. They should be specific enough to differentiate the content from the broader categories.
Make sure you output in json with double quotes.

## EXAMPLE
Here are a few examples. Do not use this specific example in your response, it is solely illustrative.

Input Paragraph:  
 ```
"This study employed a mixed-methods approach to investigate the impact of social media usage on political polarization among young adults in urban areas. 
Quantitative data was collected through a survey of 500 participants, while qualitative data was gathered via semi-structured interviews with a subset of 25 participants. 
The findings suggest a correlation between increased exposure to ideologically homogeneous content online and heightened political polarization."
 ```

Hypothetical Output from this Example (Again, illustrative and not to be used in the actual response):
```json
{"categories": ["Political Science", "Social Media Studies", "Communication Studies", "Sociology, Digital Culture"],
 "keywords": ["social media", p"olitical polarization", "young adults", "mixed-methods", "urban areas", "quantitative data", "qualitative data"],
 "tags": ["online behavior", "echo chambers", "survey methodology", "semi-structured interviews", "political communication", "digital ethnography", "ideology"]
}
 ```

## INSTRUCTIONS
1. Your response should be clearly organized, using bullet points or numbered lists to separate the categories, keywords, and tags.
2. Be precise and avoid overly broad or generic terms.
3. Prioritize terms that are commonly used within the relevant academic field.
4. Focus on accurate representation of the content provided.
5. Ensure that categories, keywords, and tags are directly relevant to the specific area of expertise you are embodying.
6. Please analyze the following paragraph and provide your expert recommendations:

"""

topics_prompt = """You are a sophisticated academic scholar with expertise in {domain}. 
You are renowned for your ability to grasp the key topics and ideas of research papers which are significant and insightful.

## TASK
Analyze the provided paragraphs and identify the key academic topics discussed.  For each topic, generate a JSON object containing the following:

*   `topic`: A precise and information-rich name for the topic. This should be as specific as possible, potentially combining multiple concepts to accurately reflect the nuanced discussion in the text.  (e.g., 'Application of Transformer Networks to Machine Translation', 'Impact of Multi-Headed Self-Attention on Long-Range Dependency Capture in Transformers').
*   `description`: A concise, general definition of the topic (1-2 sentences). Imagine you are explaining it to a colleague *unfamiliar* with the specific paper, but familiar with AI/NLP in general.  Keep the definition broad enough to encompass the general concept, even if the topic name is very specific.
*   `summary`: A detailed summary (7-10 sentences) of the topic's treatment *within the provided text*. This should include:
    *   The specific arguments made about the topic.
    *   Any evidence or examples the authors use related to the topic.
    *   The authors' conclusions or claims regarding the topic.
    *   Any limitations or critiques of the topic presented by the authors.
    *   Any comparisons to other related concepts or methods.

Output your entire response as a single, valid JSON object. The highest level should be a list called 'topics'.


## EXAMPLE
Example (using a hypothetical excerpt about Transformer Networks):

```json
{
  "topics": [
    {
      "topic": "Performance Advantages of Transformer Networks over RNNs in Machine Translation Tasks",
      "description": "This topic broadly concerns the comparison of Transformer networks and Recurrent Neural Networks (RNNs) in the context of machine translation, focusing on the superior performance characteristics of Transformers.",
      "summary": "The provided text focuses on the significant performance advantages of Transformer networks over traditional Recurrent Neural Network (RNN) based models in machine translation tasks. It argues, based on presented empirical evidence, that Transformers achieve higher BLEU scores, indicating better translation quality, across multiple language pairs and datasets.  The authors specifically attribute this superior performance to the self-attention mechanism within Transformers, which allows for more effective capture of long-range dependencies in the input text compared to the sequential processing inherent in RNNs. The text cites experimental results demonstrating faster training times for Transformers due to their parallelizable architecture, contrasting this with the inherent sequential bottleneck of RNNs.  While acknowledging the potential computational cost of Transformers for extremely long sequences, the authors downplay this limitation in the context of typical machine translation scenarios. They further support their claims by comparing Transformers to convolutional models, arguing for the greater suitability of attention mechanisms for natural language processing. The paper concludes that the shift from recurrent to attention-based models, exemplified by Transformers, represents a major advancement in the field of machine translation. The authors mention, but do not extensively analyze, the limitations imposed by dataset size on the Transformer performance."
    },
    {
        "topic": "Role of Multi-Headed Scaled Dot-Product Self-Attention in Enhancing Contextual Understanding within Transformer Networks",
        "description": "This topic encompasses the specific type of self-attention (scaled dot-product) and its multi-headed variant used in Transformer networks, and how these mechanisms contribute to the model's ability to understand context within input sequences.",
        "summary": "The provided paragraphs delve into the critical role of multi-headed scaled dot-product self-attention in enhancing contextual understanding within Transformer networks. It explains that self-attention allows each word in a sentence to attend to all other words, including itself, to derive a context-aware representation. The scaled dot-product mechanism is presented as a computationally efficient way to calculate attention weights, preventing issues that can arise with large dot products. The text emphasizes the significance of the 'multi-headed' aspect, where multiple self-attention operations are performed in parallel, each learning different aspects of the relationships between words.  This allows the model to capture diverse contextual nuances, such as syntactic and semantic dependencies, simultaneously. The authors argue that this multi-headed approach is crucial for capturing the richness of human language. They contrast this with simpler attention mechanisms, highlighting the ability of multi-headed attention to learn multiple 'representation subspaces'.  The text provides a brief mathematical overview of the scaled dot-product calculation, reinforcing its efficiency and effectiveness. The authors posit that without multi-headed attention, the Transformer's ability to model complex language structures would be significantly diminished. They conclude by highlighting the importance for future works, such as model interpretability and analysis."
    }
  ]
}
```
"""
