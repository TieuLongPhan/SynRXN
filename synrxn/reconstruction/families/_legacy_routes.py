"""Normalize the optional historical route references."""

import hashlib
import json

from synrxn.reconstruction.expansion.storage import encoded, observation


def paroutes(ctx):
    for kind in ("n1", "n5"):
        name = f"ref_routes_{kind}.json"
        if not ctx.has(name):
            continue

        def rows(name=name, kind=kind):
            routes = json.loads(ctx.path(name).read_text())
            for i, route in enumerate(routes):
                target = route.get("smiles")
                if not target:
                    raise ValueError("route root has no target SMILES")
                yield observation("C09", kind, i, "target:" + target,
                                  inputs={"target_smiles": target, "stock_id": f"PaRoutes-2.0-{kind}"},
                                  targets={"route": route}, context={"set": kind, "reference_route_hash": hashlib.sha256(encoded(route).encode()).hexdigest()},
                                  evidence="patent_extracted_reference_route")
        ctx.emit(kind, rows(), unit="reference route", assets=[name], tasks={
            "multistep_planning": {"inputs": ["target_smiles", "stock_id"], "targets": ["route"]}},
            qualification="The n1 and n5 sets overlap. Entire target groups stay together across sets; stock and reaction-exclusion files must accompany planning evaluation.")

