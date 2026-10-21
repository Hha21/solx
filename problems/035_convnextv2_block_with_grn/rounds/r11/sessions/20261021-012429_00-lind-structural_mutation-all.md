# Design session 20261021-012429_00-lind-structural_mutation-all

Task: {"round": "r11"}


### probe_b200 (1): STEP 2: bulk reduce add.f32 vs scalar red.add


### probe_b200 (2): STEP 2b: warm prefilled out, parent drain pattern


### probe_b200 (3): check /tmp persistence


### probe_b200 (4): persistence test write


### probe_b200 (5): 


### probe_b200 (6): write part 1 of kernel.cu


### probe_b200 (7): 


### probe_b200 (8): syntax check of part 2a (no build)


### compile_b200 (draft 1): r11 E1 v1: one-MMA GEMM2 + whole-image bulk-reduce drain at 7x7 + unit tables


### run_tests (draft 2): r11 E1 v1 run_tests: one-MMA + bulk-reduce whole-image units at 7x7


### probe_b200 (9): check cpp_extension build path in probe


### probe_b200 (10): noop (sanity)


### probe_b200 (11): write head part (testing persistence within one container)


### probe_b200 (12): write head again; next probe checks if it persists within the session


### probe_b200 (13): 


### probe_b200 (14): write tc_kernel body (stamped) for probe builds


### probe_b200 (15): build 4 tc-only variants (new/parent x stamped)


### probe_b200 (16): STEP1 stamps: parent-eq vs new phase C, + interleaved x3


### probe_b200 (17): v2 edits: MP=2 units at 32,768, ring staging on last unit, wait_group.read at exit


### probe_b200 (18): v2 probe: MP=2 at 32,768 + ring staging + wait.read; correctness + stamps + interleaved x3


### probe_b200 (19): table variants MP=2 at 16,768 (KS6) and 8,768 (KS12)


### probe_b200 (20): confirm PC block text for the port


### run_tests (draft 3): r11 E1 v2 final: one-MMA + whole-image bulk drain + MP=2 at 32,768 + ring staging + wait.read


### probe_b200 (21): STACK check of the probe build of the shipped tc_kernel code


### compile_b200 (draft 4): check whether compile_b200 surfaces ptxas -v output (stack frame) — placeholder


### probe_b200 (22): noop guard


### probe_b200 (23): stack check prelude
