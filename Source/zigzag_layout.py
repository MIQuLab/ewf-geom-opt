"""Zigzag physical layout for the LUCJ ansatz on IBM heavy-hex devices.

Imported from ``Code_for_SQD_incorporation/Quantum_Sampling/zigzag_layout.py``
and bundled here so the SQD solver (:mod:`sqd_solver`) can drive the
quantum-sampling step via :mod:`sqd_quantum_sampling` without an external
package.  The zigzag pattern uses two parallel alpha/beta linear chains and
connecting qubits between them; the implementation below scores all
isomorphic physical mappings to the backend's coupling graph by 2Q gate and
measurement error so the lowest-noise layout is selected.
"""

import copy

import rustworkx
from qiskit.providers import BackendV2
from rustworkx import NoEdgeBetweenNodes, PyGraph
from typing import Sequence
IBM_TWO_Q_GATES = {"cx", "ecr", "cz"}


def create_linear_chains(num_orbitals: int) -> PyGraph:
    """In zig-zag layout, there are two linear chains (with connecting qubits between
    the chains). This function creates those two linear chains: a rustworkx PyGraph
    with two disconnected linear chains. Each chain contains `num_orbitals` number
    of nodes, i.e., in the final graph there are `2 * num_orbitals` number of nodes.

    Args:
        num_orbitals (int): Number orbitals or nodes in each linear chain. They are
            also known as alpha-alpha interaction qubits.

    Returns:
        A rustworkx.PyGraph with two disconnected linear chains each with `num_orbitals`
            number of nodes.
    """
    G = rustworkx.PyGraph()

    for n in range(num_orbitals):
        G.add_node(n)

    for n in range(num_orbitals - 1):
        G.add_edge(n, n + 1, None)

    for n in range(num_orbitals, 2 * num_orbitals):
        G.add_node(n)

    for n in range(num_orbitals, 2 * num_orbitals - 1):
        G.add_edge(n, n + 1, None)

    return G


def create_lucj_zigzag_layout(
    num_orbitals: int,
    backend_coupling_graph: PyGraph,
    alpha_beta_indices: list[tuple[int, int]],
) -> tuple[PyGraph, list[tuple[int, int]]]:
    """This function creates the complete zigzag graph that 'can be mapped' to a IBM QPU with
    heavy-hex connectivity (the zigzag must be an isomorphic sub-graph to the QPU/backend
    coupling graph for it to be mapped).
    The zigzag pattern includes both linear chains (alpha-alpha interactions) and connecting
    qubits between the linear chains (alpha-beta interactions).

    Args:
        num_orbitals (int): Number of orbitals, i.e., number of nodes in each alpha-alpha linear chain.
        backend_coupling_graph (PyGraph): The coupling graph of the backend on which the LUCJ ansatz
            will be mapped and run.  See module docstring for graph-preparation notes.
        alpha_beta_indices (list): Desired alpha-beta interaction pairs (will be trimmed from the
            end if the backend cannot accommodate the full list).

    Returns:
        G_new (PyGraph): The graph with IBM backend compliant zigzag pattern.
        alpha_beta_indices (list): Final list of accommodated alpha-beta indices.
    """
    isomorphic = False
    G = create_linear_chains(num_orbitals=num_orbitals)

    while not isomorphic:
        G_new = copy.deepcopy(G)

        if not alpha_beta_indices:
            break

        # add new nodes and edges
        for i, (a, b) in enumerate(sorted(alpha_beta_indices, key=lambda x: x[0])):
            new_node = 2 * num_orbitals + i
            G_new.add_node(new_node)
            G_new.add_edge(a, new_node, None)
            G_new.add_edge(new_node, b + num_orbitals, None)
        isomorphic = rustworkx.is_subgraph_isomorphic(
            backend_coupling_graph,
            G_new,
            id_order=False,
            induced=False,
        )

        if not isomorphic:
            print(
                f"Backend cannot accomodate alpha_beta_incides {alpha_beta_indices}.\n "
                f"Removing interaction {alpha_beta_indices[-1]}"
            )
            del alpha_beta_indices[-1]

    return G_new, alpha_beta_indices


def _make_backend_cmap_pygraph(
    backend: BackendV2,
    thresh_two_q: float,
    thresh_meas: float,
) -> PyGraph:
    props = backend.properties()
    two_q_gate_name = IBM_TWO_Q_GATES.intersection(
        backend.configuration().basis_gates
    ).pop()
    graph = copy.deepcopy(backend.coupling_map.graph)

    if not graph.is_symmetric():
        graph.make_symmetric()
    backend_coupling_graph = graph.to_undirected()

    edge_list = backend_coupling_graph.edge_list()
    removed_edge = []
    for edge in edge_list:
        if set(edge) in removed_edge:
            continue
        try:
            backend_coupling_graph.remove_edge(edge[0], edge[1])
            removed_edge.append(set(edge))
        except NoEdgeBetweenNodes:
            pass

    # remove bad nodes
    node_indices = backend_coupling_graph.node_indices()
    for node_id in node_indices:
        re = props.readout_error(node_id)
        if re >= thresh_meas:
            backend_coupling_graph.remove_node(node_id)
            print(f"  >> removing node: {node_id} | Readout err: {re:0.4f}")

    edge_list = backend_coupling_graph.edge_list()
    print(f"num edges after node removal: {len(edge_list)}")

    # remove bad edges
    edge_list = backend_coupling_graph.edge_list()
    print(f"num edges before: {len(edge_list)}")
    for edge in edge_list:
        ge = props.gate_error(two_q_gate_name, edge)
        if ge >= thresh_two_q:
            backend_coupling_graph.remove_edge(edge[0], edge[1])
    edge_list = backend_coupling_graph.edge_list()
    print(f"num edges after: {len(edge_list)}")

    return backend_coupling_graph


def lightweight_layout_error_scoring(
    backend: BackendV2,
    virtual_edges: Sequence[Sequence[int]],
    physical_layouts: Sequence[int],
    two_q_gate_name: str,
) -> list[list[list[int], float]]:
    """Lighweight and heuristic function to score isomorphic layouts. There can be many zigzag patterns,
    each with different set of physical qubits, that can be mapped to a backend. Some of them may
    include less noise qubits and couplings than others. This function computes a simple error score
    for each such layout. It sums up 2Q gate error for all couplings in the zigzag pattern (layout) and
    meaurement of errors of physical qubits in the layout to compute the error score.

    Returns:
        scores (list): A list of [layout, error_score] pairs sorted in ascending order of error score.
    """
    props = backend.properties()
    scores = []
    for layout in physical_layouts:
        total_2q_error = 0
        for edge in virtual_edges:
            physical_edge = (layout[edge[0]], layout[edge[1]])
            try:
                ge = props.gate_error(two_q_gate_name, physical_edge)
            except Exception:
                ge = props.gate_error(two_q_gate_name, physical_edge[::-1])
            total_2q_error += ge
        total_measurement_error = 0
        for qubit in layout:
            meas_error = props.readout_error(qubit)
            total_measurement_error += meas_error
        scores.append([layout, total_2q_error + total_measurement_error])

    return sorted(scores, key=lambda x: x[1])


def get_zigzag_physical_layout(
    num_orbitals: int,
    backend: BackendV2,
    expected_alpha_beta_indices: list[tuple[int, int]] | None,
    thresh_two_q: float = 1.0,
    thresh_meas: float = 0.10,
) -> tuple[list[int], list[tuple[int, int]]]:
    """The main function that generates the zigzag pattern with physical qubits that can be used
    as an ``intial_layout`` in a preset passmanager/transpiler.

    Args:
        num_orbitals (int): Number of orbitals.
        backend (BackendV2): A backend.
        expected_alpha_beta_indices (list): User-defined arbitrary alpha-beta interactions.
            Due to HW limitations, the full ``expected`` list of interactions may not be
            accomodated. In that case, interaction pair from the end of the list is removed
            one-by-one. Thus, a user must order the list in a descending order of priority.
        thresh_two_q (float): Removes edges from the backend coupling graph that has
            ``2Q gate error >= thresh_two_q``. Default: 1.0 (faulty edge).
        thresh_meas (float): Removes nodes from the backend coupling graph that has
            ``measurement error >= thresh_meas``. Default: 0.10.

    Returns:
        A tuple of device compliant layout (list[int]) with zigzag pattern and the list of
        alpha-beta interaction pairs actually mapped onto the backend.
    """
    backend_coupling_graph = _make_backend_cmap_pygraph(
        backend=backend,
        thresh_two_q=thresh_two_q,
        thresh_meas=thresh_meas,
    )

    if expected_alpha_beta_indices is None:
        # Sensible default: connect alpha and beta chains every 4th orbital, fitting
        # the heavy-hex connectivity used by IBM Eagle/Heron devices.
        expected_alpha_beta_indices = [
            (p, p) for p in range(0, num_orbitals, 4)
        ]

    G, alpha_beta_qubits = create_lucj_zigzag_layout(
        num_orbitals=num_orbitals,
        backend_coupling_graph=backend_coupling_graph,
        alpha_beta_indices=expected_alpha_beta_indices,
    )
    edges = list(G.edge_list())
    num_alpha_beta_qubits = len(alpha_beta_qubits)

    if num_alpha_beta_qubits == 0:
        raise RuntimeError("No alpha-beta interaction can be accomodated. Terminating.")
    isomorphic_mappings = rustworkx.vf2_mapping(
        backend_coupling_graph,
        G,
        subgraph=True,
        id_order=False,
        induced=False,
    )

    layouts = []
    for mapping in isomorphic_mappings:
        initial_layout: list[None | int] = [None] * (
            2 * num_orbitals + num_alpha_beta_qubits
        )
        for key, value in mapping.items():
            initial_layout[value] = key
        layouts.append(initial_layout)

    if not layouts:
        raise RuntimeError("No layout found.")

    two_q_gate_name = IBM_TWO_Q_GATES.intersection(backend.configuration().basis_gates).pop()

    scores = lightweight_layout_error_scoring(
        backend=backend,
        virtual_edges=edges,
        physical_layouts=layouts,
        two_q_gate_name=two_q_gate_name,
    )
    print(f"The best layout has a total error of: {scores[0][1]}")
    # Return the alpha-alpha qubits (without the connecting ones at the tail) and
    # the list of alpha-beta pairs actually mapped onto the backend.
    return scores[0][0][:-num_alpha_beta_qubits], alpha_beta_qubits
