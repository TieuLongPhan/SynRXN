FlowER hydrogen-only disagreements
===================================

All six exact-input groups come from the train portion of FlowER. They contain
12 exported AAM records. Each pair has one heavy-atom ITS class and two
non-isomorphic explicit-H ITS classes. These are source atom-tracking
differences, not proof of chemical errors.

All six inputs are accepted in the heavy-atom AAM comparison table. The H-only
differences are retained as mechanism/proton-tracking annotations and are not
review-list exclusions.

The identifiers below are r_id values from the exported FlowER AAM table,
not original text-file line numbers. Source parent groups are the group IDs
recorded in inputs/train.txt.gz inside Data/dumps/flower.dump.zip.

.. list-table:: Source pairs
   :header-rows: 1

   * - Exported r_id pair
     - Prefix steps
     - FlowER parent groups
   * - 3496 / 3498
     - 1 / 1
     - 109633 / 109634
   * - 13814 / 18720
     - 4 / 6
     - 82205 / 82201
   * - 22032 / 22034
     - 9 / 9
     - 83454 / 83455
   * - 22033 / 22035
     - 9 / 9
     - 83454 / 83455
   * - 22544 / 22546
     - 10 / 10
     - 83454 / 83455
   * - 22545 / 22547
     - 10 / 10
     - 83454 / 83455

Concrete proton-transfer difference
------------------------------------

Records 13814 and 18720 start with the same represented atom correspondences.
H69 begins on the alcohol oxygen O23; H83 begins on the acid oxygen O36.

* The 4-step prefix (13814, source lines 592949, 592951, 592952, 592950)
  first transfers H69 from O23 to N27. H69 remains on the nitrogen and H83
  ends on the methanol oxygen O20.
* The 6-step prefix (18720, source lines 592938, 592939, 592943, 592940,
  592941, 592942) first transfers H83 from O36 to the ester oxygen O2.
  H69 eventually ends on methanol O20, while H83 ends on nitrogen N27.

Thus the source pathways explicitly encode different proton-transfer routes
with equivalent heavy-atom endpoint mappings. Keep these alternatives in
mechanism-specific data; heavy-atom AAM equivalence alone does not distinguish
their histories. Four other groups involve 9/10-step prefixes from the same
two Pd-containing parent groups, 83454 and 83455. Their original paths and
MC identifiers remain available for inspection.

Source witnesses and reproduction
----------------------------------

hydrogen-only-indices.csv contains all 12 r_id values, source parent groups,
center IDs and source_lines in the original pathway order. Line numbers are
one-based, as verified from the source reader's enumerate(text, 1).

hydrogen-only-pathways.jsonl.gz contains each complete raw elementary-step
pathway and its exported mapped endpoint reference. It pins both the FlowER
dump checksum and the unchanged source table checksum. The source-group IDs
were checked against every extracted raw line. The inspection uses
synrxn.reconstruction.builders.flower_center_groups.source_lines(dump, rows),
which reads only the requested line indices from the dump. After building the
unique AAM pool, hydrogen_conflicts.csv.gz provides the source record selection
and indices needed to repeat this inspection.
