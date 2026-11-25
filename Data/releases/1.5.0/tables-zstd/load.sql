CREATE OR REPLACE TEMP VIEW orderly_condition AS
SELECT * FROM read_csv(['orderly_condition-01.csv.zst', 'orderly_condition-02.csv.zst', 'orderly_condition-03.csv.zst'], header=true, all_varchar=true, delim=',', quote='"', escape='"');

CREATE OR REPLACE TEMP VIEW openexp AS
SELECT * FROM read_csv(['openexp.csv.zst'], header=true, all_varchar=true, delim=',', quote='"', escape='"');

CREATE OR REPLACE TEMP VIEW uspto_yield AS
SELECT * FROM read_csv(['uspto_yield.csv.zst'], header=true, all_varchar=true, delim=',', quote='"', escape='"');
