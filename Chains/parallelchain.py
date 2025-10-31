from langchain_community import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

model_1=Ollama(model="mistral")

model_2=Ollama(model="gemma3:4b")

prompt_1=PromptTemplate(
    template="Generate short and simple notes from the following text:\n{text}",
    input_variables=["text"]
)

prompt_2=PromptTemplate(
    template="Generate 5 short question ansers from the following text \n {text}",
    input_variables=["text"]
)

prompt_3=PromptTemplate(
    template="Merge the provided notes and quiz into a single documnet \n notes -> {notes} and quiz -> {quiz}",
    input_variables=['notes', 'quiz']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes_chain':prompt_1 | model_1 | parser,
    'quiz_chain':prompt_2 | model_2 | parser
})

merge_chain=prompt_3 | model_1 | parser

# Run the parallel chains
chain=parallel_chain | merge_chain

text="""Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It tries to find the best boundary known as hyperplane that separates different classes in the data. It is useful when you want to do binary classification like spam vs. not spam or cat vs. dog.

The main goal of SVM is to maximize the margin between the two classes. The larger the margin the better the model performs on new and unseen data.


Key Concepts of Support Vector Machine
Hyperplane: A decision boundary separating different classes in feature space and is represented by the equation wx + b = 0 in linear classification.
Support Vectors: The closest data points to the hyperplane, crucial for determining the hyperplane and margin in SVM.
Margin: The distance between the hyperplane and the support vectors. SVM aims to maximize this margin for better classification performance.
Kernel: A function that maps data to a higher-dimensional space enabling SVM to handle non-linearly separable data.
Hard Margin: A maximum-margin hyperplane that perfectly separates the data without misclassifications.
Soft Margin: Allows some misclassifications by introducing slack variables, balancing margin maximization and misclassification penalties when data is not perfectly separable.
C: A regularization term balancing margin maximization and misclassification penalties. A higher C value forces stricter penalty for misclassifications.
Hinge Loss: A loss function penalizing misclassified points or margin violations and is combined with regularization in SVM.
Dual Problem: Involves solving for Lagrange multipliers associated with support vectors, facilitating the kernel trick and efficient computation.
"""
res=chain.invoke({'text':text})
print(res)