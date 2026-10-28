# Graded submissions referenced in this repository

All numbers come from the author's own grader telemetry: the evaluation JSON of each submission, participant `antoniu_gugu9`, team "Antoniu". No other team's data was read. Each file sha256 was computed from the local build file. In every row it equals the sha256 in the `manifest.json` of the package uploaded for that submission, built 0.1 to 43 s before the submission was created. The grader ran flopscope 0.12.3 and whestbench 0.16.1 for every row.

Columns:

- **score:** the graded adjusted score, i.e. the mean over the 50 public MLPs of final-layer MSE x max(0.1, FLOPs/B).
- **raw:** the public-50 final-layer MSE.
- **F/B:** the public-50 mean FLOP fraction of B = 2^41.
- **F/B all:** the mean over all 100 graded MLPs.
- **ok:** MLPs completed out of 100.
- **resid / wall:** the maximum per-MLP residual time and predict wall time, in seconds.

| ID | Build | Role | Created (UTC) | File sha256 (prefix) | score | raw | F/B | F/B all | ok | resid | wall | score vs G4 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| #336780 | s43 | base of the three graded controls | 2026-10-17 19:01 | `fa8118608705` | 2.143761e-09 | 1.60690e-08 | 0.1334 | 0.1329 | 100 | 0.310 | 94.9 | x1.0594 |
| #336911 | s43cH | graded control: s43 with HZ off | 2026-10-18 08:38 | `f0afc8efab6f` | 2.474890e-09 | 1.97377e-08 | 0.1254 | 0.1249 | 100 | 0.276 | 82.5 | x1.2230 |
| #336912 | s43cL | graded control: s43 with LOT off | 2026-10-18 08:39 | `34eb9e196746` | 3.180332e-09 | 2.18140e-08 | 0.1458 | 0.1455 | 100 | 0.245 | 83.6 | x1.5716 |
| #336913 | s43cD | graded control: s43 with all D21 rows exact | 2026-10-18 08:39 | `a2c9b8b90e7d` | 2.315736e-09 | 1.60205e-08 | 0.1446 | 0.1442 | 100 | 0.290 | 90.1 | x1.1444 |
| #336845 | s44 | s43 + PEEL; slow grader worker tripped the HZ wall skip | 2026-10-18 01:20 | `007062a3c87d` | 2.181202e-09 | 1.66103e-08 | 0.1315 | 0.1311 | 100 | 0.392 | 118.5 | x1.0779 |
| #336883 | s43g | s43 + RGUARD (never fired) | 2026-10-18 05:26 | `691b5da01a3f` | 2.144898e-09 | 1.60766e-08 | 0.1334 | 0.1330 | 100 | 0.311 | 93.8 | x1.0599 |
| #336986 | s43gw | s43g + WGUARD ladder (never fired) | 2026-10-18 20:09 | `dd46f7c8626d` | 2.146324e-09 | 1.60876e-08 | 0.1334 | 0.1329 | 100 | 0.316 | 95.4 | x1.0606 |
| #336999 | s43f | minor additions (Mehler m=3 terms, wall guard, ...) | 2026-10-18 22:29 | `5b4748314145` | 2.136809e-09 | 1.61234e-08 | 0.1325 | 0.1321 | 100 | 0.326 | 98.8 | x1.0559 |
| #337102 | A3 | reference build for the forced guard tests | 2026-10-19 11:25 | `44f7d706e100` | 2.122123e-09 | 1.58465e-08 | 0.1339 | 0.1335 | 100 | 0.324 | 95.6 | x1.0487 |
| #337103 | fd1 | fire drill WG_FORCE=12:1 (guard did not arm) | 2026-10-19 11:25 | `675c0847de90` | 2.122681e-09 | 1.58510e-08 | 0.1339 | 0.1335 | 100 | 0.329 | 91.6 | x1.0490 |
| #337104 | fd2 | fire drill WG_FORCE=15:2 (guard did not arm) | 2026-10-19 11:26 | `b61f37361e7e` | 2.122988e-09 | 1.58541e-08 | 0.1339 | 0.1335 | 100 | 0.318 | 93.5 | x1.0491 |
| #337114 | fd3 | fire drill WG2 perf_counter guard (armed; test only) | 2026-10-19 12:18 | `c19ec25604ff` | 2.688240e-08 | 2.31774e-07 | 0.1159 | 0.1156 | 100 | 0.278 | 79.1 | x13.2844 |
| #337120 | V | nominated as the wall-time-safe option (call-count trims + wall-clock guard) | 2026-10-19 12:53 | `a6d76c478678` | 2.139703e-09 | 1.58761e-08 | 0.1348 | 0.1343 | 100 | 0.311 | 90.3 | x1.0574 |
| #337159 | S' | best score at the time | 2026-10-19 15:31 | `cd749acff12c` | 2.096724e-09 | 1.58111e-08 | 0.1326 | 0.1322 | 100 | 0.342 | 103.7 | x1.0361 |
| #337231 | S'' | best score at the time | 2026-10-20 00:00 | `c412d6886af4` | 2.089649e-09 | 1.58732e-08 | 0.1316 | 0.1312 | 100 | 0.348 | 104.0 | x1.0326 |
| #337317 | NLHZ | S'' + non-linear HZ head v2 | 2026-10-20 10:59 | `54c5bfd43ff8` | 2.044325e-09 | 1.55155e-08 | 0.1318 | 0.1313 | 100 | 0.334 | 96.5 | x1.0102 |
| #337322 | G2 | NLHZ + TERMITE exact cuts (99/100: one TIME_EXHAUSTED) | 2026-10-20 11:41 | `cdc84ed9fab9` | 2.036822e-09 | 1.55164e-08 | 0.1313 | 0.1306 | 99 | 0.379 | 114.2 | x1.0065 |
| #337411 | G3 | G2 chain + v5D head | 2026-10-20 21:48 | `376084b3bc40` | 2.030444e-09 | 1.54638e-08 | 0.1313 | 0.1308 | 100 | 0.332 | 96.8 | x1.0034 |
| #337608 | G4 | G3 + final-loss fine-tuned head (RELEASED FILE) | 2026-10-21 19:14 | `e71018ffa112` | 2.023603e-09 | 1.54130e-08 | 0.1313 | 0.1308 | 100 | 0.332 | 102.1 | x1.0000 |
| #337632 | G5 | G4 with head retrained on extra data; nominated | 2026-10-21 21:48 | `4d1ccf002628` | 2.022992e-09 | 1.54092e-08 | 0.1313 | 0.1308 | 100 | 0.355 | 107.9 | x0.9997 |

Notes:

- **#337608 (G4)** is the released file. G3 #337411 and G5 #337632 differ from it only in the `_NLHZ_P` line (head weights): G4/G3 score x0.99663, G5/G4 x0.99970. G5 has a wall maximum of 107.9 s on identical code; this reflects a slower grader run.
- **#336911-#336913** are the graded single-variable controls on s43 #336780 (see `controls/`). Their ratios against s43 are: HZ off, score x1.1545 (raw x1.2283); LOT off, score x1.4835 (raw x1.3575); all D21 rows exact, score x1.0802 (raw x0.9970).
- **#337103 and #337104** are fire drills of the client-context wall guard (`WG_FORCE` set). Their per-MLP FLOPs match #337102, so the guard did not arm on the grader.
- **#337114** is a test-only drill of a `perf_counter` guard (WG2), which did arm. Its score reflects the forced degraded path. WG2 is not in the released file.
- **#337322 (G2)** completed 99/100. One MLP hit the 120 s round-trip limit during a slow-worker episode on the grader.
- **#336845 (s44)** lost its HZ correction on 16 MLPs. A slow grader worker passed the `HZ_TMAX` time guards (`docs/READING_GUIDE.md` section 6).
- **Ratio columns** are computed from the unrounded telemetry values.

Full sha256 of each graded file:

| ID | Build | sha256 | package manifest created (UTC) |
|---|---|---|---|
| #336780 | s43 | `fa8118608705e45c0bad9feadf4414f56bdf84925b92910827f60e9d1ab7fe74` | 2026-10-17 19:01:09 |
| #336911 | s43cH | `f0afc8efab6f7bb1979b05eb9d1c689c5aaab74eacc29767ffd1ab306f94f8b6` | 2026-10-18 08:38:47 |
| #336912 | s43cL | `34eb9e196746091d91db26bdbf855a7e3d1bfc134bf07115703413857987e15a` | 2026-10-18 08:39:07 |
| #336913 | s43cD | `a2c9b8b90e7d061c3cd4ba06d6487d3a6f7d8baa2f3d4d40e53c73cb05db73b8` | 2026-10-18 08:39:27 |
| #336845 | s44 | `007062a3c87d11a108c93845632f5c0adf0fb1fcd7f6dcb0ab8a661f9c951668` | 2026-10-18 01:20:21 |
| #336883 | s43g | `691b5da01a3fcc6020929930875dc33a88a5f60afdd960e9d51c4058485680fc` | 2026-10-18 05:26:18 |
| #336986 | s43gw | `dd46f7c8626d6696d3fb86cd3b775b1ff43d178c9ca7511bb93e8eb577681c81` | 2026-10-18 20:09:19 |
| #336999 | s43f | `5b4748314145851f1150e6cba0b4043319272256c958d04caf6ae2029c93be88` | 2026-10-18 22:29:22 |
| #337102 | A3 | `44f7d706e100a5ce88b344e55241e39025887ee01783d0f265676a95f4ba8626` | 2026-10-19 11:25:40 |
| #337103 | fd1 | `675c0847de90cfa20bdbeb414b54586b3d873097c9e6173e4705333f4c72f838` | 2026-10-19 11:25:55 |
| #337104 | fd2 | `b61f37361e7e94a62a16b43cfeedad62d8a3df944546625b112fac12bf987328` | 2026-10-19 11:26:10 |
| #337114 | fd3 | `c19ec25604ffa5732b7e50b7b37ca69e12a88ac6975f2c7f2b0846113e717f12` | 2026-10-19 12:18:46 |
| #337120 | V | `a6d76c4786786f357f40ce2c55e0b2ddceef3d7206e374b85ed7c6512f6aba5a` | 2026-10-19 12:53:24 |
| #337159 | S' | `cd749acff12c4856cbbe8198e0303b0182ee2f77956559befd85c9145cf36817` | 2026-10-19 15:31:54 |
| #337231 | S'' | `c412d6886af46aa60ef4c80828275cb1419b59834dffa8eb2501c065c5dc560f` | 2026-10-20 00:00:46 |
| #337317 | NLHZ | `54c5bfd43ff8442359ca435bf8e8f99659ab1ffd37df40bf105b0c13e2f110d0` | 2026-10-20 10:59:47 |
| #337322 | G2 | `cdc84ed9fab99d689e605e5740bbae97e6a4c9bce6719eaa7a94e9ad4df84b03` | 2026-10-20 11:40:32 |
| #337411 | G3 | `376084b3bc40380c4c70d7565e9ade9056af946bc736f63c35001b24b2beb7c8` | 2026-10-20 21:47:53 |
| #337608 | G4 | `e71018ffa112bc2bc4274798c850fe34170462a88b320522229d8b03c001271f` | 2026-10-21 19:14:07 |
| #337632 | G5 | `4d1ccf0026284f229f7a8cfad35d0d00f306d2ee30a561889c60b78d81b818e6` | 2026-10-21 21:48:33 |

## Earlier builds cited in the write-up

From the same telemetry; file hashes are omitted because these files are not part of the release. FLOPs/B is the mean over all 100 graded MLPs.

| ID | Build | Created (UTC) | score | raw | F/B all | ok | what changed |
|---|---|---|---:|---:|---:|---:|---|
| #335600 | s01 | 2026-10-12 18:45 | 3.378533e-09 | 1.87232e-08 | 0.1798 | 100 | first graded build (LOT, demand-row D21, late (3,1) response, product reuse) |
| #335749 | s08 | 2026-10-13 10:36 | 3.148852e-09 | 1.71681e-08 | 0.1829 | 100 | HZ introduced |
| #336286 | s26 | 2026-10-15 17:40 | 2.342022e-09 | 1.61890e-08 | 0.1442 | 100 | LOT at K = 12 |
| #336456 | s31 | 2026-10-16 09:09 | 2.263973e-09 | 1.62221e-08 | 0.1390 | 100 | one complete-QR call per network (QRV = 1) becomes the default |
| #336614 | s37 | 2026-10-17 00:33 | 2.207065e-09 | 1.62927e-08 | 0.1349 | 100 | intermediate build; used for a reproducibility measurement (E05-04) |
| #336728 | s40 | 2026-10-17 13:11 | 2.160193e-09 | 1.61919e-08 | 0.1329 | 100 | Edgeworth means for dropped rows |
| #336749 | s42 | 2026-10-17 15:32 | 2.143468e-09 | 1.60655e-08 | 0.1329 | 100 | marg2 rule |
| #336773 | s42r | 2026-10-17 18:03 | 2.146094e-09 | 1.60856e-08 | 0.1329 | 100 | s42 with call-count trims, identical local predictions; base of the D² halving step to s43 |
