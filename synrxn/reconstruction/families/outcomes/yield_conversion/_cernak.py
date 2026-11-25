"""Normalize the Cernak ORD coupling experiments."""

from synrxn.reconstruction.expansion.storage import observation


def ord_cernak(ctx):
    name = "cernak_cn.parquet"
    if not ctx.has(name):
        return
    import pyarrow.parquet as pq
    from google.protobuf.json_format import MessageToDict
    from ord_schema.proto.reaction_pb2 import Reaction

    def rows():
        ordinal = 0
        for batch in pq.ParquetFile(ctx.path(name)).iter_batches(batch_size=1024):
            for raw in batch.to_pylist():
                proto = Reaction()
                proto.ParseFromString(raw["reaction"])
                row = MessageToDict(proto, preserving_proto_field_name=True)
                components = [c for value in row.get("inputs", {}).values() for c in value.get("components", [])]
                reactants = sorted({v["value"] for c in components if c.get("reaction_role") == "REACTANT"
                                    for v in c.get("identifiers", []) if v.get("type") == "SMILES"})
                if not reactants:
                    raise ValueError("ORD reaction has no identified reactants")
                products, measurements, times = [], [], []
                for outcome in row.get("outcomes", []):
                    if "reaction_time" in outcome:
                        times.append(outcome["reaction_time"])
                    for product in outcome.get("products", []):
                        if product.get("is_desired_product"):
                            products.extend(v["value"] for v in product.get("identifiers", []) if v.get("type") == "SMILES")
                            measurements.extend(m for m in product.get("measurements", []) if m.get("type") == "YIELD")
                yields = [m["percentage"]["value"] for m in measurements if "percentage" in m and "value" in m["percentage"]]
                targets = {"yield_measurements": measurements}
                if len(yields) == 1:
                    targets["assay_yield_pct"] = yields[0]
                inputs = {"reactants": reactants, "desired_products": products,
                          "components": components, "conditions": row.get("conditions", {}),
                          "reaction_time": times}
                yield observation("C03", "cn_screen", ordinal, ".".join(reactants), inputs=inputs, targets=targets,
                                  native=row, context={"ord_reaction_id": raw["reaction_id"],
                                                       "dataset_id": "ord_dataset-805ad863feef48579d95d86a728035f4"})
                ordinal += 1
    fields = ["reactants", "desired_products", "components", "conditions", "reaction_time"]
    ctx.emit("cn_screen", rows(), unit="HTE experiment", assets=[name], tasks={
        key: {"inputs": fields, "targets": ["assay_yield_pct"]}
        for key in ["yield_conversion", "condition_ranking", "sequential_optimization"]},
        units={"assay_yield_pct": "percent (assay yield)"},
        qualification="UPLC assay yields are selected by ORD measurement type and desired-product flag. Area and internal-standard measurements remain native metadata.")

