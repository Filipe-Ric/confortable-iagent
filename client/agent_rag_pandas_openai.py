from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.documents import Document
import pandas as pd
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Definir o estado do agente
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Ferramenta Pandas para análise de sensores
@tool
def analyze_sensor_data(csv_path: str, metric: str) -> str:
    """Analisa dados de sensores em um CSV e retorna métricas solicitadas."""
    try:
        df = pd.read_csv(csv_path)
        if metric == "mean_vibration":
            result = df["vibration"].mean()
            return f"Média de vibração: {result:.2f} mm/s"
        elif metric == "max_temperature":
            result = df["temperature"].max()
            return f"Temperatura máxima: {result:.2f} °C"
        else:
            return "Métrica não suportada."
    except Exception as e:
        return f"Erro ao analisar dados: {str(e)}"

# Ferramenta RAG para consultar manuais
@tool
def search_manual(query: str) -> str:
    """Busca informações em um manual técnico usando RAG."""
    # Simular manual com documentos
    documents = [
        Document(page_content="Limite seguro de vibração: 4.5 mm/s. Acima disso, risco de falha iminente.", metadata={"source": "manual"}),
        Document(page_content="Temperatura máxima recomendada: 70°C. Acima disso, desligar equipamento.", metadata={"source": "manual"}),
    ]
    # Criar banco vetorial FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    # Buscar documentos relevantes
    results = vectorstore.similarity_search(query, k=1)
    if results:
        return results[0].page_content
    return "Nenhuma informação encontrada no manual."

# Configurar ferramentas
tools = [analyze_sensor_data, search_manual]

# Configurar LLM
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
llm_with_tools = llm.bind_tools(tools)

# Função do chatbot
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Construir o grafo
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Definir arestas
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compilar o grafo com memória
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Exemplo de uso
if __name__ == "__main__":
    # Simular dados de sensores
    sensor_data = pd.DataFrame({
        "vibration": [3.2, 4.8, 5.1, 2.9],
        "temperature": [65.0, 72.0, 68.0, 70.0]
    })
    sensor_data.to_csv("sensors.csv", index=False)

    # Configurar thread para memória
    config = {"configurable": {"thread_id": "1"}}

    # Interagir com o agente
    question = "Qual é o risco de falha com base nos dados de sensores?"
    response = graph.invoke({"messages": [{"role": "user", "content": question}]}, config)
    print(response["messages"][-1].content)