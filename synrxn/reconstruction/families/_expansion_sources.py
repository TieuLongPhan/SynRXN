"""Select family recipes by expansion source ID."""

from synrxn.reconstruction.families.outcomes.yield_conversion._minot import borylation
from synrxn.reconstruction.families.outcomes.yield_conversion._cernak import ord_cernak
from synrxn.reconstruction.families.selectivity.product_distribution._flow_panels import catechol
from synrxn.reconstruction.families.selectivity.enantioselectivity._orgaicat import orgaicat
from synrxn.reconstruction.families.physics.activation_barrier._rdb19 import rdb19
from synrxn.reconstruction.families._legacy_routes import paroutes
from synrxn.reconstruction.families.physics.reaction_rates._quantumpioneer_expansion import (
    quantumpioneer, undirected_reaction,
)


ADAPTERS = {"C02": borylation, "C03": ord_cernak, "C06": catechol, "C07": rdb19,
            "C09": paroutes, "C11": quantumpioneer, "C16": orgaicat}

__all__ = ['ADAPTERS', 'undirected_reaction']
