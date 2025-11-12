import numpy as np
import random
import networkx as nx
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D

class AgentsAggregation:
    def __init__(self, agents_num, malicious_num,shuffle=False) -> None:
        assert agents_num > malicious_num
        self.agents_num = agents_num
        self.malicious_num = malicious_num
        self.legitimate_nums=agents_num-malicious_num
        self.labels = ["L"] * self.legitimate_nums + ["M"] * self.malicious_num
        if shuffle:
            random.shuffle(self.labels)
        self.agents_list = [Agent(i, agents_num, self.labels[i], self) for i in range(agents_num)]
        self.create_network()

    def create_network(self):
        # Initialize directed graph
        G = nx.DiGraph()
        G.add_nodes_from(range(self.agents_num))  # Add all agents (legitimate and malicious) as nodes

        # Get the indices of legitimate ('L') and malicious ('M') agents based on self.labels
        legitimate_indices = [i for i, label in enumerate(self.labels) if label == 'L']
        malicious_indices = [i for i, label in enumerate(self.labels) if label == 'M']

        # Create a directed cyclic graph for legitimate agents
        for i in range(len(legitimate_indices)):
            G.add_edge(legitimate_indices[i], legitimate_indices[(i + 1) % len(legitimate_indices)])

        # Add additional random directed edges among legitimate agents
        additional_edges = int(len(legitimate_indices) * 0.1)
        for _ in range(additional_edges):
            u, v = random.sample(legitimate_indices, 2)
            G.add_edge(u, v)

        # Ensure the subgraph GL induced by the legitimate agents is strongly connected
        while not nx.is_strongly_connected(G.subgraph(legitimate_indices)):
            u, v = random.sample(legitimate_indices, 2)
            G.add_edge(u, v)

        # Explicitly connect malicious agents to legitimate agents with probability 0.7
        for i in legitimate_indices:
            for j in malicious_indices:
                if random.random() < 0.7:
                    if random.random() < 0.5:
                        G.add_edge(i, j)  # Directed edge from legitimate to malicious
                    else:
                        G.add_edge(j, i)  # Directed edge from malicious to legitimate

        # Convert to adjacency matrix
        self.net = nx.to_numpy_array(G)


    def visualize_subgraph(self, subgraph):
        # Define node colors and shapes based on labels
        color_map = ['skyblue' for _ in subgraph.nodes()]
        
        # Define node positions using spring layout for better visual separation
        pos = nx.spring_layout(subgraph)  # Use spring layout for better separation
        
        # Draw the subgraph nodes
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(subgraph, pos, node_color=color_map, node_size=600, edgecolors='black')
        
        # Draw directed edges with arrows
        nx.draw_networkx_edges(subgraph, pos, arrowstyle='->', arrowsize=20, alpha=0.5)
        
        # Add labels
        nx.draw_networkx_labels(subgraph, pos, font_color='white', font_size=10)
        
        # Display the plot
        plt.title('Subgraph of Legitimate Agents (In Progress)', fontsize=16)
        plt.axis('off')
        plt.show()
    
    def visualize_network(self,show_fig=False, save_path=None):
        # Create a directed graph from the adjacency matrix
        G = nx.from_numpy_array(self.net, create_using=nx.DiGraph)
        
        # Define node colors and shapes based on labels
        color_map = ['skyblue' if label == 'L' else 'lightgreen' for label in self.labels]
        node_shapes = {'L': 'o', 'M': 's'}
        
        # Define node positions using circular layout for better visual separation
        pos = nx.circular_layout(G)  # Use circular layout for clear visualization
        
        # Draw legitimate and malicious agents with different shapes
        plt.figure(figsize=(12, 12))
        for label in node_shapes:
            nodes = [i for i in range(len(self.labels)) if self.labels[i] == label]
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_color=[color_map[i] for i in nodes],
                node_shape=node_shapes[label],
                node_size=600,
                edgecolors='black'
            )
        
        # Draw directed edges with arrows
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, alpha=0.5)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_color='white', font_size=10)
        
        # Display the plot
        plt.title('Directed Network Visualization with Legitimate (Skyblue) and Malicious (Lightgreen) Agents', fontsize=16)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, format='jpeg', dpi=300, bbox_inches='tight')
        if show_fig: 
            plt.show()
        

    def visualize_network_3d(self,show_fig=False, save_path=None, k_value=0.3):
        # Create a directed graph from the adjacency matrix
        G = nx.from_numpy_array(self.net, create_using=nx.DiGraph)

        # Define node colors based on labels
        color_map = ['skyblue' if label == 'L' else 'lightgreen' for label in self.labels]

        # Use a 3D spring layout for node positions, and control node density using the `k` parameter
        pos = nx.spring_layout(G, dim=3, seed=42, k=k_value)  # Increase k_value to space out nodes
        
        # Extract node positions into lists
        Xn = [pos[k][0] for k in G.nodes()]  # x-coordinates of nodes
        Yn = [pos[k][1] for k in G.nodes()]  # y-coordinates of nodes
        Zn = [pos[k][2] for k in G.nodes()]  # z-coordinates of nodes

        # Create trace for edges
        edge_trace = []
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_trace.append(go.Scatter3d(
                x=[x0, x1, None], y=[y0, y1, None], z=[z0, z1, None],
                mode='lines',
                line=dict(color='black', width=2),
                hoverinfo='none'
            ))

        # Create trace for nodes
        node_trace = go.Scatter3d(
            x=Xn, y=Yn, z=Zn,
            mode='markers',
            marker=dict(
                symbol='circle',
                size=8,
                color=color_map,  # Node colors
                line=dict(color='black', width=1)
            ),
            text=[f'Node {i}' for i in range(len(self.labels))],  # Hover text
            hoverinfo='text'
        )

        # Create the 3D plot layout with fixed aspect ratio for aligned axes
        layout = go.Layout(
            title="3D Directed Network Visualization",
            showlegend=False,
            scene=dict(
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False),
                aspectratio=dict(x=1, y=1, z=1),  # Set equal aspect ratio for all axes
                aspectmode='manual'
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            hovermode='closest'
        )

        # Plot the network graph
        fig = go.Figure(data=edge_trace + [node_trace], layout=layout)

        # Save or display
        if save_path:
            fig.write_image(save_path)
        if show_fig:
            fig.show()


    def visualize_network_3d_to_2d(self, show_fig=False, save_path=None, k_value=10, elev=100, azim=100):
        # Create a directed graph from the adjacency matrix
        G = nx.from_numpy_array(self.net, create_using=nx.DiGraph)

        # Define node colors based on labels ('L' -> Legitimate, 'M' -> Malicious)
        color_map = ['skyblue' if label == 'L' else 'lightgreen' for label in self.labels]

        # Define node shapes based on labels
        node_shapes = {'L': 'o', 'M': 's'}  # Circle for 'L', Square for 'M'

        # Use a 3D spring layout for node positions, and control node density using the `k` parameter
        pos = nx.spring_layout(G, dim=3, seed=42, k=k_value)  # Increase k_value to space out nodes

        # Create 3D plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Extract node positions into lists
        Xn = [pos[k][0] for k in G.nodes()]  # x-coordinates of nodes
        Yn = [pos[k][1] for k in G.nodes()]  # y-coordinates of nodes
        Zn = [pos[k][2] for k in G.nodes()]  # z-coordinates of nodes

        # Plot edges
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            ax.plot([x0, x1], [y0, y1], [z0, z1], color='black', alpha=0.5)

        # Plot nodes with different shapes for legitimate and malicious agents
        for label, shape in node_shapes.items():
            nodes = [i for i, lbl in enumerate(self.labels) if lbl == label]
            ax.scatter(
                [Xn[i] for i in nodes],  # x-coordinates of selected nodes
                [Yn[i] for i in nodes],  # y-coordinates of selected nodes
                [Zn[i] for i in nodes],  # z-coordinates of selected nodes
                c=[color_map[i] for i in nodes],  # Node colors based on label
                s=100,
                edgecolors='black',
                depthshade=True,
                marker=shape  # Different shapes for legitimate ('L') and malicious ('M')
            )

        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)  # Adjust elevation and azimuth for better viewing

        # Hide axes
        ax.set_axis_off()

        # Save as 2D image
        if save_path:
            plt.savefig(save_path, format='jpeg', dpi=300, bbox_inches='tight')
        if show_fig:
            plt.show()

    def reset_opinions(self):
        for agent in self.agents_list:
            agent.reset_opinion()

    def get(self, i):
        return self.agents_list[i]
        
    def initial_observation(self,rounds=30):
        for _ in trange(rounds):
            for agent in self.agents_list:
                agent.update_opinions()

class Agent:
    def __init__(self, agent_id, nums, label, aggregation):
        self.agent_id = agent_id
        self.label = label
        self.nums = nums
        self.opinions = np.ones(nums)  # Trust opinions, initially 0
        self.opinions_prev=np.ones(nums)
        self.N_in=None
        self.N_out=None
        self.beta = np.zeros(nums)  # Aggregated trust values, initially 0
        self.xi = np.random.uniform(-50, 50)  # Initialize xi within [-50, 50]
        self.zi = self.xi
        self.si = 0
        self.aggregation = aggregation

    def reset_opinion(self):
        self.opinions = np.ones(self.nums)  # Trust opinions, initially 0
        self.opinions_prev = np.ones(self.nums)

    def determine_neighborhoods(self):
        # N_in = []
        # N_out = []
        # for j in range(self.nums):
        #     if self.aggregation.net[self.agent_id][j] == 1:  # Determine neighbors
        #         if self.opinions[j] >= 0.5:  # Trust threshold
        #             N_out.append(j)

        # for i in range(self.nums):
        #     if self.aggregation.net[i][self.agent_id] == 1:  # Determine neighbors
        #         if self.opinions[i] >= 0.5:  # Trust threshold
        #             N_in.append(i)

        # self.N_in=N_in.copy()
        # self.N_out=N_out.copy()
        return self.N_in, self.N_out

    def update_opinions(self):
        if self.label=="L":
            self.opinions_prev=self.opinions.copy()
            # N^{in}_i
            for j in range(self.nums):
                if self.aggregation.net[j][self.agent_id] == 1:  # 有j->i，即N^{in}_i
                    alpha_ij = self.stochastic_observation(self.aggregation.get(j))
                    self.beta[j] += (alpha_ij - 0.5)
                    self.opinions[j] = 1 if self.beta[j] >= 0 else 0
            # N^{out}_i
            for j in range(self.nums):
                if self.aggregation.net[self.agent_id][j] == 1:  # 有i->j，即N^{out}_i
                    alpha_ij = self.stochastic_observation(self.aggregation.get(j))
                    self.beta[j] += (alpha_ij - 0.5)
                    self.opinions[j] = 1 if self.beta[j] >= 0 else 0
                
            # 确定可信入邻接点集合 N_in_i[k]
            self.N_in = [i for i in range(self.nums) if self.aggregation.net[i][self.agent_id] == 1 and self.opinions[i] >= 0.5]

            self.N_out=[j for j in range(self.nums) if self.aggregation.net[self.agent_id][j] == 1 and self.opinions[j] >= 0.5]

            # 确定非N^{in}_i
            for q in range(self.nums):
                if self.aggregation.net[q][self.agent_id] == 0:  # 没有q->i
                    opinion_sum = sum([self.aggregation.get(j).opinions_prev[q] for j in range(self.nums) if self.aggregation.net[j][self.agent_id] == 1])
                    N_in_sum=len(self.N_in)
                    if N_in_sum>0:
                        self.opinions[q] = opinion_sum / N_in_sum
                    else:
                        self.opinions[q] = 0  # 或者你可以定义其他默认值
            
            # 确定非N^{out}_i
            for q in range(self.nums):
                if self.aggregation.net[self.agent_id][q] == 0:  # 没有i->q
                    # 出邻接点opinion和
                    opinion_sum = sum([self.aggregation.get(j).opinions_prev[q] for j in range(self.nums) if self.aggregation.net[self.agent_id][j] == 1])
                    N_out_sum=len(self.N_out)
                    if N_out_sum>0:
                        self.opinions[q] = opinion_sum / N_out_sum
                    else:
                        self.opinions[q] = 0  # 或者你可以定义其他默认值

            # self.N_in += [j for j in range(self.nums) if self.aggregation.net[j][self.agent_id] == 0 and self.opinions[j] >= 0.5]
            # # 确定可信出邻接点集合 N_out_i[k]
            # self.N_out += [j for j in range(self.nums) if self.aggregation.net[self.agent_id][j] == 0 and self.opinions[j] >= 0.5]
        else:
            self.N_in=[i for i,w in enumerate(self.aggregation.net[:,self.agent_id]) if w>0]
            self.N_out=[i for i,w in enumerate(self.aggregation.net[self.agent_id,:]) if w>0]


    def compute_coefficients(self):
        W_in = np.ones(len(self.N_in)) / len(self.N_in)  # R
        W_out = np.ones(len(self.N_out)) / len(self.N_out)  #C
        return W_in, W_out

    def stochastic_observation(self, agent):
        # Generate stochastic trust observation αij[k]
        if agent.label == "M":
            return random.uniform(0.25, 0.65)
        else:
            return random.uniform(0.35, 0.75)

    
    def data_process(self,data,enable=True,max_val=50,min_val=-50):
        if enable and self.label=="M":
            return np.where(data>(max_val+min_val)/2,max_val,min_val)
        else:
            return data


if __name__ == "__main__":
    η = 0.1  # Optimization parameter
    λ = 0.5  # Lazy update parameter
    rounds = 30  # Initial observation window iterations
    # agents_num = 8  # Total agents
    # malicious_num = 3  # Malicious agents

    agents_num = 50  # Total agents
    malicious_num = 20  # Malicious agents

    aggregation = AgentsAggregation(agents_num, malicious_num)
    # aggregation.visualize_network(save_path=r"C:\Users\chen\Desktop\pic.jpeg")
    # aggregation.visualize_network_3d(save_path=r"C:\Users\chen\Desktop\pic_3d.jpeg")
    aggregation.visualize_network_3d_to_2d(show_fig=True,save_path=r"C:\Users\chen\Desktop\pic_3d_t0_2d.jpeg")
    aggregation.initial_observation()

    # for _ in range(rounds):
    #     for agent in aggregation.agents_list:
    #         agent.update_opinions()
    #         N_in, N_out = agent.determine_neighborhoods()
    #         Cji, Rij = agent.compute_coefficients()
    #         si_neighbors = [aggregation.get(j).si for j in N_in]
    #         zj_neighbors = [aggregation.get(j).zi for j in N_out]
    #         gradient = -50 if agent.label == "M" and agent.xi > 0 else 50


    #         agent.lazy_update(η, λ)

    print("")