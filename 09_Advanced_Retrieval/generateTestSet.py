import glob
import os
from typing import List

from datasets import Dataset
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.graph import NodeType
from ragas.testset.persona import Persona
from datasets import Dataset
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    default_query_distribution,
)
from ragas.testset.transforms import (
    HeadlineSplitter,
    HeadlinesExtractor,
    KeyphrasesExtractor,
    Parallel,
    TitleExtractor,
    Transforms,
    apply_transforms,
)

os.environ["LANGSMITH_PROJECT"] = "AIE8/09_Advanced_Retrieval"

# define your LLM and Embedding Model
# here we are using the same LLM and Embedding Model that we used to generate the testset
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

transformer_llm = generator_llm
embedding_model = generator_embeddings

# trans = default_transforms(documents=docs, llm=transformer_llm, embedding_model=embedding_model)
    #  Extract information that can be used to generate relationships between nodes in the knowledge graph
TRANSFORMS = [
    # Parallel(       
    HeadlinesExtractor(llm=transformer_llm),
    HeadlineSplitter(max_tokens=1200, min_tokens=500),
    TitleExtractor(llm=transformer_llm),
    KeyphrasesExtractor(llm=transformer_llm),
    # ),
]
RUN_CONFIG = RunConfig(max_workers=32, timeout=120)
KG_PATH = os.path.join("evals", "react", "react_reference_knowledge_graph.json")
TESTSET_PATH = os.path.join("evals", "react", "react_reference_testset.keyphrases.headlines.csv")
EVALUATION_DATASET_PATH = os.path.join("evals", "react", "react_reference_evaluation_dataset.keyphrases.headlines.csv")
# Create custom personas
persona1 = Persona(
    name="Experienced Web Developer",
    role_description="An experienced web developer with 5+ years of experience in frontend development, familiar with JavaScript, HTML, CSS, and various frameworks. Looking to deepen their understanding of React's advanced concepts and best practices. Background: Has worked with multiple JavaScript frameworks and libraries, understands modern web development patterns, and wants to master React's ecosystem and advanced features."
)

persona2 = Persona(
    name="JavaScript Framework Beginner", 
    role_description="A developer new to JavaScript frameworks with basic JavaScript knowledge. May not be familiar with modern web development concepts, component-based architecture, or advanced JavaScript features. Background: Has basic JavaScript knowledge but limited experience with frameworks, libraries, or modern web development patterns. Needs clear explanations of fundamental concepts and step-by-step guidance."

)

# Create persona list
PERSONA_LIST = [persona1, persona2]

def generate_knowledge_graph():
    kg = KnowledgeGraph()
    docs: List[Document] = []
    markdown_files = glob.glob("data/react/*.md")
    for md_file in markdown_files:
        with open(md_file, "r", encoding="utf-8") as f:
            node = Node(
                type=NodeType.DOCUMENT,
                properties={ 
                    "page_content": f.read(),
                    "document_metadata": {
                        "source": md_file,
                        "title": os.path.basename(md_file),
                        "author": "React.js Documentation",
                        "url": f"https://react.dev/reference/react/{os.path.basename(md_file)}",
                        "format": "markdown"
                    }
                }
            )
            doc = Document(page_content=node.properties["page_content"], metadata=node.properties["document_metadata"])
            kg.nodes.append(node)
            docs.append(doc)
    return kg, docs

# Create a dataset of documents
# dataset = Dataset.from_dict({"items": documents})

def apply_kg_transforms(kg: KnowledgeGraph, transforms: Transforms, rc: RunConfig):
    return apply_transforms(kg, transforms, rc)

def load_kg(input_path: str):
    kg = KnowledgeGraph.load(input_path)
    return kg

def save_kg(kg: KnowledgeGraph, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    kg.save(output_path)

# FIRST RUN - Generate the knowledge graph and apply the initial transforms

# rc = RunConfig(max_workers=32, timeout=120)
# kg, docs = generate_knowledge_graph()
# apply_kg_transforms(kg, transforms, rc)
# save_kg(kg, os.path.join("evals", "react", "react_reference_knowledge_graph_v2.json"))

# Second run - load the knowledge graph, apply more transforms (optional)

def generate_testset():
    if not os.path.exists(KG_PATH):
        kg, docs = generate_knowledge_graph()
        apply_kg_transforms(kg, TRANSFORMS, RUN_CONFIG)
        save_kg(kg, KG_PATH) 
    else:
        kg = load_kg(KG_PATH)

    # Test Set Generation - DIY
    # dataset = generator.generate_with_langchain_docs(docs, testset_size=10)
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings, knowledge_graph=kg, persona_list=PERSONA_LIST)

    # query_distribution = default_query_distribution(generator_llm)
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="headlines",), 0.6),
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="keyphrases"), 0.4),
        # (SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="title"), 0.1),
    ]

    testset = generator.generate(testset_size=10, query_distribution=query_distribution, run_config=RUN_CONFIG, num_personas=2)

    testset.to_csv(TESTSET_PATH)
    evaluation_dataset = testset.to_evaluation_dataset()
    evaluation_dataset.to_csv(EVALUATION_DATASET_PATH)

#### OTHER UTILITIES
def get_react_docs():
    docs: List[Document] = []
    markdown_files = glob.glob("data/react/*.md")
    for md_file in markdown_files:
        with open(md_file, "r", encoding="utf-8") as f:

            doc = Document(page_content=f.read(), metadata={
                "source": md_file,
                "title": os.path.basename(md_file),
                "author": "React.js Documentation",
                "url": f"https://react.dev/reference/react/{os.path.basename(md_file)}",
                "format": "markdown"
            })
            docs.append(doc)
    return docs

def format_dataset(chain, df):
    """
    Evaluate a chain by pre-computing answers and contexts, following the working pattern.
    """
    evaluation_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    
    for idx, row in df.iterrows():
        question = row["user_input"]
        ground_truth = row["reference"]
        
        print(f"   [{idx + 1}/{len(df)}] {question[:60]}...")
        
        try:
            # Call the chain
            result = chain.invoke({"question": question})
            
            # Extract answer and contexts from chain output
            answer = result.get("response", {}).content if hasattr(result.get("response", {}), 'content') else str(result.get("response", ""))
            contexts = result.get("context", [])
            
            # Convert contexts to list of strings if needed
            if isinstance(contexts, list):
                contexts = [str(ctx.page_content) if hasattr(ctx, 'page_content') else str(ctx) for ctx in contexts]
            else:
                contexts = [str(contexts)]
            
            # Store for evaluation
            evaluation_data["question"].append(question)
            evaluation_data["answer"].append(answer)
            evaluation_data["contexts"].append(contexts)
            evaluation_data["ground_truth"].append(ground_truth)
            
        except Exception as e:
            print(f"      ⚠ Error: {e}")
            # Add empty entries to maintain alignment
            evaluation_data["question"].append(question)
            evaluation_data["answer"].append("")
            evaluation_data["contexts"].append([])
            evaluation_data["ground_truth"].append(ground_truth)
    
    # Create dataset from dict (matching working example)
    dataset = Dataset.from_dict({
        "question": evaluation_data["question"],
        "answer": evaluation_data["answer"],
        "contexts": evaluation_data["contexts"],
        "ground_truth": evaluation_data["ground_truth"],
    })
    
    return dataset
