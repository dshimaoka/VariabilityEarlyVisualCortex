import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def compute_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def create_surface_graph(faces, vertices):
    G = nx.Graph()

    # Add nodes for each vertex
    for vertex in vertices:
        G.add_node(tuple(vertex), pos=vertex)

    # Add edges based on the faces
    for face in faces:
        for i in range(len(face)):
            for j in range(i + 1, len(face)):
                G.add_edge(tuple(vertices[face[i]]), tuple(vertices[face[j]]), weight=compute_distance(vertices[face[i]], vertices[face[j]]))

    return G

def create_volume_graph(faces, vertices):
    G = nx.Graph()

    # Add nodes for each vertex
    for vertex in vertices:
        G.add_node(tuple(vertex), pos=vertex)

    # Add edges based on the faces
    for face in faces:
        for i in range(len(face)):
            for j in range(i + 1, len(face)):
                G.add_edge(tuple(vertices[face[i]]), tuple(vertices[face[j]]), weight=compute_distance(vertices[face[i]], vertices[face[j]]))

    return G

def create_and_connect_volume_faces(volume_faces, vertices, G):
    # Connect vertices of each face to form edges within the volume
    for face in volume_faces:
        for i in range(len(face)):
            for j in range(i + 1, len(face)):
                G.add_edge(tuple(vertices[face[i]]), tuple(vertices[face[j]]), weight=compute_distance(vertices[face[i]], vertices[face[j]]))

def compute_minimal_path(faces, volume_faces, vertices, start_point, end_point):
    G = create_volume_graph(faces, vertices)
    create_and_connect_volume_faces(volume_faces, vertices, G)

    # Use Dijkstra's algorithm to find the shortest path
    path = nx.shortest_path(G, source=tuple(start_point), target=tuple(end_point), weight='weight')

    # Compute the traveling distance of the path
    distance = sum(G.edges[path[i], path[i + 1]]['weight'] for i in range(len(path) - 1))

    return path, distance

# ver0: only computes path on the surface
# def compute_minimal_path(faces, vertices, start_point, end_point):
#     G = create_surface_graph(faces, vertices)

#     # Use Dijkstra's algorithm to find the shortest path
#     path = nx.shortest_path(G, source=tuple(start_point), target=tuple(end_point), weight='weight')

#     # Compute the traveling distance of the path
#     distance = sum(G.edges[path[i], path[i + 1]]['weight'] for i in range(len(path) - 1))

#     return path, distance


def create_convex_surface(num_faces=200, radius=1):
    # Generate random points on the unit sphere
    phi = np.linspace(0, np.pi, num_faces)
    theta = np.linspace(0, 2 * np.pi, num_faces)
    phi, theta = np.meshgrid(phi, theta)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # Reshape to obtain 1D arrays
    vertices = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

    # Define faces
    faces = []
    for i in range(num_faces):
        for j in range(num_faces):
            p1 = i * num_faces + j
            p2 = (i + 1) * num_faces + j
            p3 = (i + 1) * num_faces + (j + 1)
            p4 = i * num_faces + (j + 1)

            # Ensure that indices do not exceed the total number of vertices
            p1 %= num_faces**2
            p2 %= num_faces**2
            p3 %= num_faces**2
            p4 %= num_faces**2

            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])

    return faces, vertices

def plot_convex_surface(faces, vertices, start_point, end_point):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the convex surface
    poly3d = Poly3DCollection([vertices[face] for face in faces], facecolor='b', alpha=0.5)
    ax.add_collection3d(poly3d)

    # Plot start and end points
    ax.scatter(start_point[0], start_point[1], start_point[2], c='g', s=100, label='Start Point')
    ax.scatter(end_point[0], end_point[1], end_point[2], c='r', s=100, label='End Point')

    # Compute and plot the minimal path
    path, _ = compute_minimal_path(faces, vertices, start_point, end_point)
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], c='y', linewidth=2, label='Minimal Path')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point2) - np.array(point1))

def compute_volume_faces(faces, vertices):
    # Find the center of the convex hull of the vertices
    center = np.mean(vertices, axis=0)

    # Create volume faces by connecting each vertex to the center
    volume_faces = []
    for face in faces:
        volume_face = face.copy()
        volume_face.append(center)
        volume_faces.append(volume_face)

    return volume_faces

# Example usage:
num_faces = 100
radius = 1
faces, vertices = create_convex_surface(num_faces, radius)
volume_faces = compute_volume_faces(faces, vertices)

# Example start and end points
start_point = vertices[0]
end_point = vertices[939]

plot_convex_surface(faces, vertices, start_point, end_point)




path, distance = compute_minimal_path(faces, volume_faces, vertices, start_point, end_point)
print("Minimal Path:", path)
print("Traveling Distance:", distance)


distance2 = euclidean_distance(start_point, end_point)