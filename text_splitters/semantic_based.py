from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_amount=1,
    breakpoint_threshold_type="standard_deviation"
)

sample_text = """
Artificial Intelligence (AI) is the simulation of human intelligence in machines.
These machines are programmed to think and learn like humans.
AI has become one of the most transformative technologies of the 21st century.

Machine Learning is a subset of AI that allows systems to learn from data.
Instead of being explicitly programmed, ML models improve through experience.
Common ML algorithms include decision trees, neural networks, and support vector machines.

Deep Learning is a further subset of Machine Learning using neural networks with many layers.
It has revolutionized fields like image recognition and natural language processing.
Models like GPT and BERT are built on deep learning architectures.

Natural Language Processing (NLP) enables machines to understand human language.
It powers applications like chatbots, translation tools, and sentiment analysis.
NLP combines linguistics and machine learning to process text and speech.

Computer Vision allows machines to interpret and understand visual information.
It is used in self-driving cars, facial recognition, and medical imaging.
Convolutional Neural Networks (CNNs) are the backbone of most computer vision tasks.
"""

result = text_splitter.split_text(sample_text)

print((result))