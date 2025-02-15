gen_mapping_prompt = """Please help me compoase a prompt for the following task.
In the task, the AI will act as an academic scholar.
Given excerpts of sections (usually including openning setences and closing sentences), the AI would identify how each section corresponding to 

"Abstract",
"Introduction", "Background", "Introduction and Motivation", "Preliminary", 
"Related Work", "Literature Review", "Related Research",
"Methods", "Methodology", "Method", "Approach", "Work Flow", "Materials and Methods", "Computation Function", "Problem Formulation", "Mathmatical Formulation", "Psedo Code",
"Experiment", "Experiment Settings", "Experimental Results", "Evaluation", "Experiments",
"Analysis", "Results", "Findings", "Data Analysis", "Results and Findings",
"Conclusion", "Discussion", "Results and Discussion", "Further Discussion", 
"References",
"Acknowledgments", 
"FAQ", "Frequently Asked Questions",
"Implementation Code", "Examples", "Appendix"
"""

gen_keywords_prompt = """Please help me compose a prompt to complete user's request. 

In the prompt, AI would play the role of academic scholar in specific area. 
When user provide a paragraph from a research paper, the AI would carefully review it and propose suggested categories, 
keywords and tags, which would be helpful to better understand the paper and organize the knowledge.
"""

gen_topics_prompt = """Could you please help me generate a prompt to complete the following task?

In the task, the AI would be asked to play a sophisticated academic scholar in a specific domain. 
User would present the AI some paragraphs from an academic paper and ask the AI to identify the key topics in these paragraphs. 
AI would respond to user the concrete topics, description of the topics, detailed summaries based on the provided paragraph on the topics. 
The output would be in json format
"""


