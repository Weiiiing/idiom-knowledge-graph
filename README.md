# idiom-knowledge-graph
idiom knowledge graph constructed by deep learning

## 1. Introduction
- Based on transfer learning and text augmentation technology to identify Chinese idiom sentiment labels, and using the external platform to obtain the idiom knowledge to fusion to form a Chinese idiom knowledge graph.

## 2. File
- "DUTIR.xlsx" is the ontology database of emotion vocabulary of the Dalian University of Technology, which is the migration data source for sentiment classification of the project.
- "idiom_ data_ augmentation.xlsx" is an augmented text of idioms based on Chinese knowledge, including idiom interpretation, synonyms, and character-related phrases, which is used to solve the problem of low accuracy of small sample learning.
- "idiom_knowledge_base.xlsx" is a relational database of idiom knowledge graphs.

## 3. Requirement
```  
python == 3.7.10  
keras == 2.4.3  
tensorflow == 2.3.0  
sklearn == 0.24.1  
keras_bert == 0.86.0
```
