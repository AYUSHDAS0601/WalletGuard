"""
Graph Query Routes — subgraph extraction and serialization.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from api.models.schemas import GraphEdge, GraphNode, GraphQueryRequest, GraphQueryResponse

router = APIRouter(prefix="/api/v1/graph", tags=["Graph"])


def get_pipeline(request: Request):
    return request.app.state.pipeline


@router.post(
    "/query",
    response_model=GraphQueryResponse,
    summary="Extract a transaction subgraph around an address",
)
async def query_graph(
    body: GraphQueryRequest,
    pipeline=Depends(get_pipeline),
):
    """
    Return the ego-network (up to `depth` hops) around `center_address`.
    Optionally filter edges by minimum ETH weight.
    """
    address = body.center_address.lower().strip()
    if not address.startswith("0x"):
        raise HTTPException(status_code=400, detail="Invalid address format")

    try:
        # Fetch recent transactions for the center address
        df = pipeline.etherscan.get_wallet_transactions(address, limit=2000)
        if df.empty:
            return GraphQueryResponse(center_address=address, depth=body.depth)

        if "value_eth" not in df.columns and "value" in df.columns:
            df["value_eth"] = df["value"] / 1e18

        G = pipeline.graph_builder.build_networkx(df)
        sub = pipeline.graph_builder.get_subgraph(
            G, center=address, depth=body.depth, min_weight=body.min_edge_weight_eth
        )

        if sub.number_of_nodes() == 0:
            return GraphQueryResponse(center_address=address, depth=body.depth)

        # Compute risk scores for nodes (heuristic from degree)
        max_deg = max((sub.degree(n) for n in sub.nodes()), default=1)

        nodes = [
            GraphNode(
                id=n,
                risk_score=min(sub.degree(n) / max(max_deg, 1), 1.0),
                node_type="wallet",
                in_degree=sub.in_degree(n),
                out_degree=sub.out_degree(n),
            )
            for n in sub.nodes()
        ]
        edges = [
            GraphEdge(
                source=u,
                target=v,
                weight=round(d.get("weight", 0), 6),
                tx_count=d.get("tx_count", 1),
            )
            for u, v, d in sub.edges(data=True)
        ]

        return GraphQueryResponse(
            center_address=address,
            depth=body.depth,
            nodes=nodes,
            edges=edges,
            node_count=len(nodes),
            edge_count=len(edges),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
