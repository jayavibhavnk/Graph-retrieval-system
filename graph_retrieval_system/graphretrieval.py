import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from joblib import Parallel, delayed
from langchain_text_splitters import CharacterTextSplitter
from openai import OpenAI

class GraphRAG():
    def __init__(self):
        self.graph = None
        self.lines = None
        self.embeddings = None
    
    def constructGraph(self, text, similarity_threshold=0):
        text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
        )

        pre_lines = text_splitter.create_documents([text])

        lines = [i.page_content for i in pre_lines]

        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(lines)
        graph = nx.Graph()

        def add_edges(i):
            edges = []
            for j in range(i , len(lines)):
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])
                if similarity[0][0] > similarity_threshold:
                    edges.append((i, j, similarity[0][0]))
            return edges

        edge_lists = Parallel(n_jobs=-1)(delayed(add_edges)(i) for i in range(len(lines)))
        edges = [edge for edge_list in edge_lists for edge in edge_list]
        graph.add_weighted_edges_from(edges)

        self.graph = graph
        self.lines = lines
        self.embeddings = embeddings

        print("Graph Created Successfully!")
        return graph, lines, embeddings

    def create_graph_from_file(self, file, similarity_threshold = 0):
        file = open(file, 'r')
        text_data = file.read()
        self.graph, self.lines, self.embeddings = create_graph_from_text(text_data, similarity_threshold = similarity_threshold)
        print("Graph created Successfully!")
    
    def compute_similarity(self, current_node, graph, lines, query_embedding):
        similar_nodes = []
        for neighbor in graph.neighbors(current_node):
            neighbor_embedding = embeddings[neighbor]
            neighbor_similarity = cosine_similarity([query_embedding], [neighbor_embedding])[0][0]
            similar_nodes.append((neighbor, neighbor_similarity))
        return similar_nodes

    def a_star_search_parallel(self, graph, lines, embeddings, query_text, k=5):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query_text])[0]

        pq = [(0, None, 0)]
        visited = set()
        similar_nodes = []

        while pq:
            _, current_node, similarity_so_far = heapq.heappop(pq)

            if current_node is not None:
                node_text = lines[current_node]
                similar_nodes.append((node_text, similarity_so_far))

            if len(similar_nodes) >= k:
                break

            compute_similarity_partial = delayed(self.compute_similarity)
            results = Parallel(n_jobs=-1)(compute_similarity_partial(neighbor, graph, lines, query_embedding) for neighbor in (graph.neighbors(current_node) if current_node is not None else range(len(lines)-1)))

            for result in results:
                for neighbor, neighbor_similarity in result:
                    if neighbor not in visited:
                        priority = -neighbor_similarity
                        heapq.heappush(pq, (priority, neighbor, similarity_so_far + neighbor_similarity))
                        visited.add(neighbor)

        return similar_nodes

    def retrieveFromGraph(self, query):        
        similar_nodes = self.a_star_search_parallel(self.graph, self.lines, self.embeddings, query, k=5)
        l_text = []
        for node, similarity in similar_nodes:
            l_text.append(node)
            # print(f"Similarity: {similarity:.4f}, Node: {node}")
        
        return l_text
    
    def queryLLM(self, query):
        l_text = self.retrieveFromGraph(query)
        full_text = ""
        for i in l_text:
            full_text = full_text + i
        
        prompt_template = """
        You are an assistant that answers user's queries, you will be given a context and and some instruction, you will answer the query based on this, 
        context: {context},
        query: {query}
        """
        ans = self.query_openai(prompt_template.format(context = full_text, query = query))

        return ans

    def query_openai(self, query):
        client = OpenAI()
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": query}
        ],
        n = 1
        )
        return(completion.choices[0].message.content)


class KnowledgeRAG():
    def __init__(self):
        pass
    
    def constructKB(self):
        pass
    
    def setup_graphdb(self):
        pass
    
    def queryKB(self):
        pass

class GraphRetrieval(GraphRAG, KnowledgeRAG):
    def __init__():
        pass

